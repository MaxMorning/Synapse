import torch
from torch.cuda.amp import autocast as autocast
import tqdm

from trainer.BaseTrainer import BaseTrainer


class LowLightTrainer(BaseTrainer):
    def __init__(self, networks, train_loader, val_loaders_dict, losses, metrics, optimizer, resume_state, init_method, tensorboard_log_dir, options):
        super().__init__(networks, train_loader, val_loaders_dict, losses, metrics, optimizer, resume_state, init_method, tensorboard_log_dir, options)

    def train_step(self, options, iter_index, scaler):
        train_data = next(self.train_loader)

        low_input = train_data['input'].cuda()
        normal_gt = train_data['ground_truth'].cuda()

        with autocast(options['speed_up']['enable_amp']):
            output = self.network.train_forward(low_input)

            loss_supervised = self.loss_fn(
                output,
                normal_gt,
                'supervised_loss', iter_index,
                'Supervised Loss')

            loss = loss_supervised
            loss = loss / options['train']['iter_per_optim_step']

        # calc gradient and backward
        if options['speed_up']['enable_amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Grad clip
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=20, norm_type=2)

        if iter_index % options['train']['iter_per_optim_step'] == (options['train']['iter_per_optim_step'] - 1):
            if options['speed_up']['enable_amp']:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)


    def fr_eval_step(self, iter_index, options):
        self.network.eval()

        with torch.no_grad():
            eval_metric_result = {
                'iter': iter_index,
                'result': {}
            }
            for eval_set_name, eval_loader in self.val_loaders_dict.items():
                eval_metric_result['result'][eval_set_name] = {}

                eval_loader_pbar = tqdm.tqdm(eval_loader)
                metrics_result = {}
                for metric_name in self.metrics:
                    metrics_result[metric_name] = {}

                for val_data in eval_loader_pbar:
                    file_name = val_data['file_name']
                    low_input = val_data['input'].cuda()
                    normal_gt = val_data['ground_truth'].cuda()
                    eval_batch_size = low_input.shape[0]

                    with autocast(options['speed_up']['enable_amp'] or options['speed_up']['fast_eval']):
                        output = self.network.test_forward(
                            low_input
                        )
                        output = torch.clamp(output, 0, 1)

                    with autocast(options['speed_up']['enable_amp']):
                        for metric_name, metric in self.metrics.items():
                            for sample_i in range(eval_batch_size):
                                metric_result = metric(output[sample_i:sample_i + 1], normal_gt[sample_i:sample_i + 1])
                                metrics_result[metric_name][file_name[sample_i]] = metric_result

                eval_metric_result['result'][eval_set_name] = metrics_result

            # log & visualization
            self.metric_result_log_and_visual(eval_metric_result)

            # save best checkpoint
            self.check_and_save_best_checkpoint(iter_index, eval_metric_result, options)

        self.network.train()
