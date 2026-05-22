import torch
import torch.distributed as dist
import tqdm

from trainer.BaseTrainer import BaseTrainer


class LowLightTrainer(BaseTrainer):
    def __init__(self, networks, train_loaders_dict, valid_loaders_dict, losses, metrics, optimizer, resume_state, init_method, tensorboard_log_dir, options):
        super().__init__(networks, train_loaders_dict, valid_loaders_dict, losses, metrics, optimizer, resume_state, init_method, tensorboard_log_dir, options)

    def train_step(self, options, iter_index, scaler):
        train_data = next(self.train_loader)

        low_input = train_data['input'].to(self.device)
        normal_gt = train_data['ground_truth'].to(self.device)

        fsdp_mp = self._get_fsdp_mixed_precision_enabled(options)
        use_autocast = options['speed_up']['enable_amp'] and not fsdp_mp

        with torch.amp.autocast('cuda', enabled=use_autocast):
            output = self.network(low_input)

            loss_supervised = self.loss_fn(
                output,
                normal_gt,
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
                            for sample_i in range(eval_batch_size):
                                metric_result = metric(output[sample_i:sample_i + 1], normal_gt[sample_i:sample_i + 1])
                                metrics_result[metric_name][file_name[sample_i]] = metric_result

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
