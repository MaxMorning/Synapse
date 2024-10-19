from loguru import logger
from util.util import init_obj


def create_trainer(options, train_loaders_dict, valid_loaders_dict):
    """ create trainer """

    # set metrics and loss
    metric_list = [create_metric(item_opt) for item_opt in options['trainer']['which_metrics']]
    metric_name_list = [str(item_opt) for item_opt in options['trainer']['which_metrics']]
    metrics = {(metric.__class__.__name__ if metric.__class__.__name__ != 'partial' else metric_name): metric for metric, metric_name in zip(metric_list, metric_name_list)}

    losses = {}
    for loss_name, loss_content in options['trainer']['which_losses'].items():
        losses[loss_name] = [create_loss(item_opt) for item_opt in loss_content]

    # set network
    networks = {}
    for network_name, network_config in options['trainer']['which_networks'].items():
        networks[network_name] = create_network(network_config, has_grad=True)

        network_param_size = (sum(param.numel() for param in networks[network_name].parameters())) / 1e6
        logger.info("{} Network {} param {:.2f} M".format(network_name, networks[network_name].__class__.__name__, network_param_size))

    # set trainer
    trainer_opt = options['trainer']['which_trainer']
    trainer_opt['args'].update(
        {
            'networks': networks,
            'train_loaders_dict': train_loaders_dict,
            'valid_loaders_dict': valid_loaders_dict,
            'losses': losses,
            'metrics': metrics,
            'resume_state': options['path']['resume_state'],
            'tensorboard_log_dir': options['path']['tensorboard_log_dir'],
            'options': options
        }
    )

    trainer = init_obj(trainer_opt, default_file_name='trainer.trainer', init_type='trainer')

    return trainer


def create_metric(metric_opt):
    return init_obj(metric_opt, default_file_name='metric.metric', init_type='Metric')


def create_loss(loss_opt):
    return init_obj(loss_opt, default_file_name='loss.loss', init_type='Loss').to('cuda').eval()


def create_network(network_opt, has_grad):
    """ define network with weights initialization """
    net = init_obj(network_opt, default_file_name='arch.network', init_type='Network')

    if not has_grad:
        for param in net.parameters():
            param.detach_()
    return net

def create_optimizer(optimizer_opt):
    return init_obj(optimizer_opt, default_file_name='torch.optim', init_type='Adam')
