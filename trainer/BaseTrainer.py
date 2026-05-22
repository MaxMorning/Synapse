import random
import torch.distributed as dist
import functools
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from loguru import logger
import os
from PIL import Image
import time

import data
import util.util as TheUtil
import json
from torch.cuda.amp import GradScaler
import csv
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from util.Profiler import Profiler
from util.scheduler import ReproducibleScheduler


class BaseTrainer:
    def __init__(self, networks, train_loaders_dict, valid_loaders_dict, losses, metrics, optimizer,
                 resume_state, init_method, tensorboard_log_dir, options):
        # Distributed setup
        self.is_fsdp = options['distributed']['enable_fsdp'] and dist.is_initialized()
        self.is_main_process = TheUtil.is_main_process()
        local_rank = options.get('local_rank', 0)
        self.device = torch.device('cuda', local_rank)

        self.network = networks['enhance_network'].to(self.device)

        # FSDP wrapping
        if self.is_fsdp:
            self.network = self._wrap_model_fsdp(self.network, options)
            logger.info('Model wrapped with FSDP on rank {}'.format(dist.get_rank()))

        self.train_loader = data.InfinityIterator(train_loaders_dict['Paired'])
        self.valid_loaders_dict = valid_loaders_dict

        assert len(losses) > 0
        self.loss_function_dict = losses
        self.loss_recorder = {}
        self.clear_loss_recorder()

        assert options['train']['val_iter'] % options['train']['report_iter'] == 0
        assert options['train']['save_iter'] % options['train']['report_iter'] == 0

        self.metrics = metrics
        self.metric_csv_path = options['path']['metric_csv']
        if self.is_main_process:
            self.init_csv()

        optimizer['args'].update({
            "params":
                list(filter(lambda p: p.requires_grad, self.network.parameters()))
        })
        self.optimizer = TheUtil.init_obj(optimizer, default_file_name='torch.optim', init_type='Optimizer')

        # lr scheduler
        self.scheduler = ReproducibleScheduler(
            optimizer=self.optimizer,
            strategy=options['train']['lr_scheduler'],
            last_epoch=options['train']['n_iter'] // options['train']['report_iter']
        )
        logger.success('LR scheduler is created in {} method.'.format(options['train']['lr_scheduler']))

        # resume training state
        if resume_state is None:
            self.global_step = 0

            if not self.is_fsdp:
                self.init_weights(self.network, init_type=init_method)
        else:
            self.load_from_disk(resume_state, options)

        if self.is_main_process:
            self.summery_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            self.summery_writer = None

        # init current_best_info
        self.current_best_info = {}
        for val_set_name in self.valid_loaders_dict:
            self.current_best_info[val_set_name] = {}
            for metric_name in self.metrics:
                self.current_best_info[val_set_name][metric_name] = {
                    'value': -2147483647,
                    'iter': -1
                }

        self.global_best_info = {
            'value': -2147483647,
            'iter': -1
        }

    def _wrap_model_fsdp(self, network, options):
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy, MixedPrecision, CPUOffload, BackwardPrefetch
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        dist_cfg = options['distributed']
        local_rank = options.get('local_rank', 0)

        # Sharding strategy
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        sharding_strategy = strategy_map[dist_cfg['sharding_strategy']]

        # Mixed precision policy
        mp_policy = None
        if dist_cfg['mixed_precision'] == 'bf16':
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dist_cfg['mixed_precision'] == 'fp16':
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # Auto wrap policy — wrap at TransformerBlock granularity
        from arch.Restormer.restormer_arch import TransformerBlock
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock}
        )

        # CPU offload
        cpu_offload = CPUOffload(offload_params=True) if dist_cfg['cpu_offload'] else None

        # Backward prefetch
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = prefetch_map.get(dist_cfg['backward_prefetch'])

        network = FSDP(
            network,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            forward_prefetch=dist_cfg['forward_prefetch'],
            sync_module_states=dist_cfg['sync_module_states'],
            use_orig_params=dist_cfg['use_orig_params'],
            device_id=local_rank,
        )

        # Activation checkpointing
        if dist_cfg['activation_checkpointing']:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                apply_activation_checkpointing,
                CheckpointImpl,
            )
            apply_activation_checkpointing(
                network,
                check_fn=lambda submodule: isinstance(submodule, TransformerBlock),
            )
            logger.info('Activation checkpointing applied to TransformerBlock modules')

        logger.info('FSDP config: sharding={}, mixed_precision={}, cpu_offload={}, act_ckpt={}'.format(
            dist_cfg['sharding_strategy'], dist_cfg['mixed_precision'],
            dist_cfg['cpu_offload'], dist_cfg['activation_checkpointing']))

        return network

    def _get_fsdp_mixed_precision_enabled(self, options):
        return self.is_fsdp and options['distributed']['mixed_precision'] is not None

    def train(self, options):
        logger.info('Train start')
        fsdp_mp = self._get_fsdp_mixed_precision_enabled(options)

        if fsdp_mp:
            logger.info('FSDP mixed precision enabled ({}).'.format(options['distributed']['mixed_precision']))
        elif options['speed_up']['enable_amp']:
            logger.info('AMP enable.')
        else:
            logger.info('AMP disable.')

        # switch to train mode
        self.network.train()

        # When FSDP handles mixed precision, disable GradScaler
        if fsdp_mp:
            scaler = GradScaler(enabled=False)
        else:
            scaler = GradScaler(enabled=options['speed_up']['enable_amp'])
        torch.cuda.empty_cache()

        self.optimizer.zero_grad(set_to_none=True)

        with Profiler(options['path']['profile_path']) as prof:
            while self.global_step <= options['train']['n_iter']:
                logger.info('Train of iter {} start'.format(self.global_step + 1))
                logger.info('Current LearningRate {}'.format(self.scheduler.get_lr()))
                if self.summery_writer is not None:
                    self.summery_writer.add_scalar('Learning rate', self.scheduler.get_lr(), self.global_step)

                tqdm_bar = tqdm.tqdm(range(0, options['train']['report_iter']), desc='Report {}'.format(self.global_step // options['train']['report_iter']),
                                     disable=not self.is_main_process)
                if options['phase'] == 'debug':
                    # debug mode
                    for _ in tqdm_bar:
                        self.train_step(options, self.global_step, scaler)
                        self.global_step += 1

                        prof.step()

                        if self.global_step % options['train']['val_iter'] == 0 and self.global_step != 0:
                            # val part
                            logger.info('Validation of iter {} start in debug mode'.format(self.global_step))
                            self.fr_eval_step(self.global_step, options)
                            logger.info('Validation of iter {} finish '.format(self.global_step))
                else:
                    for _ in tqdm_bar:
                        self.train_step(options, self.global_step, scaler)
                        self.global_step += 1

                        prof.step()

                logger.info('Train of iter {} finish'.format(self.global_step))
                self.print_loss_per_report_iter()

                if self.global_step % options['train']['val_iter'] == 0 and self.global_step != 0:
                    # val part
                    logger.info('Validation of iter {} start'.format(self.global_step))
                    self.fr_eval_step(self.global_step, options)
                    logger.info('Validation of iter {} finish'.format(self.global_step))

                if self.global_step % options['train']['save_iter'] == 0 and self.global_step != 0:
                    self.save_everything(self.global_step, self.global_step, options)
                    logger.info('Checkpoint of iter {} saved'.format(self.global_step))

                if self.global_step >= options['train']['finish_iter']:
                    logger.info('Reach finish iter {}'.format(options['train']['finish_iter']))
                    break

                self.scheduler.adjust_lr(self.global_step // options['train']['report_iter'])

        logger.info('Train end')

    def train_step(self, options, iter_index, scaler):
        train_data = next(self.train_loader)

        frames_input = train_data['LRs'].to(self.device)
        frames_gt = train_data['HRs'].to(self.device)

        fsdp_mp = self._get_fsdp_mixed_precision_enabled(options)
        use_autocast = options['speed_up']['enable_amp'] and not fsdp_mp

        with torch.amp.autocast('cuda', enabled=use_autocast):
            output = self.network(frames_input, frames_gt)

            loss_supervised = self.loss_fn(
                torch.clamp(output, min=0, max=1),
                frames_gt,
                'supervised_loss', iter_index,
                'Supervised Loss')

            loss = loss_supervised
            loss = loss / options['train']['iter_per_optim_step']

        # calc gradient and backward
        if use_autocast:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Grad clip
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=20, norm_type=2)

        if iter_index % options['train']['iter_per_optim_step'] == (options['train']['iter_per_optim_step'] - 1):
            if use_autocast:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

    def fr_eval_step(self, iter_index, options):
        self.network.eval()

        fsdp_mp = self._get_fsdp_mixed_precision_enabled(options)
        use_autocast_eval = (options['speed_up']['enable_amp'] or options['speed_up']['fast_eval']) and not fsdp_mp
        use_autocast_metric = options['speed_up']['enable_amp'] and not fsdp_mp

        with torch.no_grad():
            eval_metric_result = {
                'iter': iter_index,
                'result': {}
            }
            for eval_set_name, eval_loader in self.valid_loaders_dict.items():
                eval_metric_result['result'][eval_set_name] = {}

                eval_loader_pbar = tqdm.tqdm(eval_loader, disable=not self.is_main_process)
                metrics_result = {}
                for metric_name in self.metrics:
                    metrics_result[metric_name] = {}

                for val_data in eval_loader_pbar:
                    file_name = val_data['file_name']
                    low_input = val_data['input'].to(self.device)
                    normal_gt = val_data['ground_truth'].to(self.device)
                    eval_batch_size = low_input.shape[0]

                    with torch.amp.autocast('cuda', enabled=use_autocast_eval):
                        output = self.network(
                            low_input
                        )
                        output = torch.clamp(output, 0, 1)

                    with torch.amp.autocast('cuda', enabled=use_autocast_metric):
                        for metric_name, metric in self.metrics.items():
                            metric_result = metric(output, normal_gt)
                            for sample_i in range(eval_batch_size):
                                metrics_result[metric_name][file_name[sample_i]] = metric_result[sample_i].item()

                eval_metric_result['result'][eval_set_name] = metrics_result

            # Gather metric results from all ranks
            if self.is_fsdp:
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, metrics_result)
                if self.is_main_process:
                    merged = {}
                    for metric_name in self.metrics:
                        merged[metric_name] = {}
                        for rank_result in gathered:
                            merged[metric_name].update(rank_result[metric_name])
                    metrics_result = merged

            # log & visualization (rank0 only)
            if self.is_main_process:
                self.metric_result_log_and_visual(eval_metric_result)
            
        self.check_and_save_best_checkpoint(iter_index, eval_metric_result, options)

        if self.is_fsdp:
            dist.barrier()
        self.network.train()

    def loss_fn(self, x, y, loss_name, iter_index, tag):
        loss_sum = 0
        for loss in self.loss_function_dict[loss_name]:
            single_loss = loss(x, y)
            loss_sum = loss_sum + single_loss * loss.weight_from_iter(iter_index)

            with torch.no_grad():
                # visualization (rank0 only)
                if self.summery_writer is not None:
                    self.summery_writer.add_scalar('[{}] {}'.format(tag, loss.__class__.__name__), single_loss.item(),
                                                   self.global_step)

        with torch.no_grad():
            if not isinstance(loss_sum, type(0)):
                loss_sum_item = loss_sum.item()
                if self.summery_writer is not None:
                    self.summery_writer.add_scalar('[{}] Total Loss'.format(tag), loss_sum_item, self.global_step)
                self.loss_recorder[loss_name].append(loss_sum_item)

        return loss_sum

    # Tool Functions
    def init_weights(self, net, init_type='kaiming', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    logger.critical(
                        'Initialization method [{}] of {} is not implemented'.format(init_type, m.__class__.__name__))
                    raise NotImplementedError(
                        'Initialization method [{}] of {} is not implemented'.format(init_type, m.__class__.__name__))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        if init_type is None:
            logger.info('{} initialized as default'.format(net.__class__.__name__))
        else:
            net.apply(init_func)
            logger.success("{} initialized as {}".format(net.__class__.__name__, init_type))

    def print_loss_per_report_iter(self):
        logger.info('Current Loss:')
        for loss_name, loss_value in self.loss_recorder.items():
            if len(loss_value) > 0:
                logger.info('\t{}: {}'.format(loss_name, np.mean(loss_value)))

        self.clear_loss_recorder()

    def clear_loss_recorder(self):
        for loss_name, loss_funcs in self.loss_function_dict.items():
            if len(loss_funcs) > 0:
                self.loss_recorder[loss_name] = []

    def save_everything(self, prefix, iter_index, options):
        if self.is_fsdp:
            self._save_everything_fsdp(prefix, iter_index, options)
        else:
            self._save_everything_local(prefix, iter_index, options)

    def _save_everything_local(self, prefix, iter_index, options):
        use_safetensors = options['path'].get('checkpoint_format', 'pth') == 'safetensors'
        network_label = self.network.__class__.__name__
        save_network(prefix, network=self.network, network_label='enhance_' + network_label,
                     save_dir=options['path']['checkpoint'], use_safetensors=use_safetensors)

        # Train info
        train_info = {
            'iter': iter_index,

            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.scheduler.get_state(),

            # random generator save
            'torch_rng': torch.get_rng_state(),
            'torch_cuda_rng': torch.cuda.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate()
        }
        torch.save(train_info, os.path.join(options['path']['checkpoint'], "{}_train_info.pth".format(prefix)))
        logger.success('{} saved'.format(prefix))

    def _save_everything_fsdp(self, prefix, iter_index, options):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        use_safetensors = options['path'].get('checkpoint_format', 'pth') == 'safetensors'
        network_label = self.network.__class__.__name__

        # Gather full state dict to rank0
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.network, StateDictType.FULL_STATE_DICT, full_state_cfg):
            state_dict = self.network.state_dict()

        if self.is_main_process:
            ext = '.safetensors' if use_safetensors else '.pth'
            save_filename = '{}_{}{}'.format(prefix, 'enhance_' + network_label, ext)
            save_path = os.path.join(options['path']['checkpoint'], save_filename)
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()

            if use_safetensors:
                from safetensors.torch import save_file
                save_file(state_dict, save_path)
            else:
                torch.save(state_dict, save_path)

            logger.info('Train info will not be saved in FSDP.')
            # # Train info
            # train_info = {
            #     'iter': iter_index,

            #     'optimizer': self.optimizer.state_dict(),
            #     'lr_scheduler': self.scheduler.get_state(),

            #     # random generator save
            #     'torch_rng': torch.get_rng_state(),
            #     'torch_cuda_rng': torch.cuda.get_rng_state(),
            #     'numpy_rng': np.random.get_state(),
            #     'python_rng': random.getstate()
            # }
            # torch.save(train_info, os.path.join(options['path']['checkpoint'], "{}_train_info.pth".format(prefix)))
            logger.success('{} saved (FSDP full state dict)'.format(prefix))

        # Synchronize after saving
        dist.barrier()

    def load_from_disk(self, resume_state, options):
        if self.is_fsdp:
            self._load_from_disk_fsdp(resume_state, options)
        else:
            self._load_from_disk_local(resume_state, options)

    def _load_from_disk_local(self, resume_state, options):
        sr_network_label = self.network.__class__.__name__
        load_network(resume_state, network=self.network, network_label='enhance_' + sr_network_label)

        # Train info
        train_info = torch.load(resume_state + "_train_info.pth")
        self.global_step = train_info['iter']

        self.optimizer.load_state_dict(train_info['optimizer'])
        self.scheduler.set_state(train_info['lr_scheduler'])

        torch.set_rng_state(train_info['torch_rng'])
        torch.cuda.set_rng_state(train_info['torch_cuda_rng'])
        np.random.set_state(train_info['numpy_rng'])
        random.setstate(train_info['python_rng'])

    def _load_from_disk_fsdp(self, resume_state, options):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        sr_network_label = self.network.__class__.__name__
        model_path = _get_model_path(resume_state, 'enhance_' + sr_network_label)

        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.network, StateDictType.FULL_STATE_DICT, full_state_cfg):
            if self.is_main_process:
                if os.path.exists(model_path):
                    if model_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(model_path)
                    else:
                        state_dict = torch.load(model_path, map_location='cpu')
                    logger.info('Loading FSDP checkpoint from [{}]'.format(model_path))
                else:
                    logger.warning('Checkpoint [{}] not found, skipping model load'.format(model_path))
                    state_dict = {}
            else:
                state_dict = {}
            self.network.load_state_dict(state_dict)

        # Train info (all ranks load)
        train_info_path = resume_state + "_train_info.pth"
        if os.path.exists(train_info_path):
            train_info = torch.load(train_info_path, map_location='cpu')
            self.global_step = train_info['iter']

            self.optimizer.load_state_dict(train_info['optimizer'])
            self.scheduler.set_state(train_info['lr_scheduler'])

            torch.set_rng_state(train_info['torch_rng'])
            torch.cuda.set_rng_state(train_info['torch_cuda_rng'])
            np.random.set_state(train_info['numpy_rng'])
            random.setstate(train_info['python_rng'])

        dist.barrier()

    def init_csv(self):
        csv_header = 'iter,'
        for val_set_name in self.valid_loaders_dict:
            for metric in self.metrics:
                csv_header += '{}_{},'.format(val_set_name, metric)

        csv_header = csv_header[:-1]
        with open(self.metric_csv_path, 'w') as csv_file:
            csv_file.write(csv_header)

    def metric_result_log_and_visual(self, metric_result_dict):
        logger.info('Eval result at iter {}:'.format(metric_result_dict['iter']))

        csv_line = '\n{},'.format(metric_result_dict['iter'])
        for dataset_name, dataset_result in metric_result_dict['result'].items():
            for metric_name, metric_dict in dataset_result.items():
                avg_metric_value = np.mean(list(metric_dict.values()))

                if self.summery_writer is not None:
                    self.summery_writer.add_scalar('[{}] {}'.format(dataset_name, metric_name),
                                                   avg_metric_value, metric_result_dict['iter'])
                csv_line += str(float(avg_metric_value)) + ','

                logger.info('\t{} at {}: {:.6f}'.format(metric_name, dataset_name, avg_metric_value))

        csv_line = csv_line[:-1]
        with open(self.metric_csv_path, 'a+') as csv_file:
            csv_file.write(csv_line)

    def check_and_save_best_checkpoint(self, iter_index, eval_metric_result, options):
        save_prefixes = []
        if self.is_main_process:
            watching_metrics = options['trainer']['watching_metrics']

            metric_sum = 0
            # discover best
            for val_set_name, val_result_info in self.current_best_info.items():
                for metric_name, metric_result_info in val_result_info.items():
                    val_set_metric_name = '{}_{}'.format(val_set_name, metric_name)
                    if val_set_metric_name not in watching_metrics:
                        continue

                    metric_sum += np.mean(list(eval_metric_result['result'][val_set_name][metric_name].values()))
                    single_better = np.mean(list(eval_metric_result['result'][val_set_name][metric_name].values()))

                    if metric_result_info['value'] < single_better:
                        # new best
                        logger.info('New Best {} under {} : {}'.format(metric_name, val_set_name, single_better))
                        self.current_best_info[val_set_name][metric_name]['value'] = float(single_better)
                        self.current_best_info[val_set_name][metric_name]['iter'] = iter_index
                        save_prefixes.append(val_set_metric_name)

            if self.global_best_info['value'] < metric_sum:
                self.global_best_info['value'] = float(metric_sum)
                self.global_best_info['iter'] = iter_index
                logger.info('New Global Best : {}'.format(metric_sum))
                save_prefixes.append('global')

        if self.is_fsdp:
            broadcast_list = [save_prefixes]
            dist.broadcast_object_list(broadcast_list, src=0)
            save_prefixes = broadcast_list[0]

        for prefix in save_prefixes:
            self.save_everything(prefix, iter_index, options)

def _get_model_path(resume_state, network_label):
    """Resolve model path, preferring .safetensors over .pth if both exist."""
    st_path = "{}_{}.safetensors".format(resume_state, network_label)
    pth_path = "{}_{}.pth".format(resume_state, network_label)
    if os.path.exists(st_path):
        return st_path
    return pth_path


def load_network(resume_state, network, network_label):
    if resume_state is None:
        return
    logger.info('Begin loading pretrained model [{:s}] ...'.format(network_label))

    model_path = _get_model_path(resume_state, network_label)

    if not os.path.exists(model_path):
        logger.warning('Pretrained model in [{:s}] is not existed, Skip it'.format(model_path))
        return

    logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))

    if model_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        network.load_state_dict(state_dict)
    else:
        network.load_state_dict(torch.load(model_path))

    logger.success('Load pretrained model {} success.'.format(network_label))


def save_network(prefix, network, network_label, save_dir, use_safetensors=False):
    """ save network structure """
    ext = '.safetensors' if use_safetensors else '.pth'
    save_filename = '{}_{}{}'.format(prefix, network_label, ext)
    save_path = os.path.join(save_dir, save_filename)

    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()

    if use_safetensors:
        from safetensors.torch import save_file
        save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)
