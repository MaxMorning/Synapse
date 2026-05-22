import os
import torch
import argparse
from loguru import logger

import util.util as TheUtil
import data as TheData
import trainer as TheTrainer


def main(main_options):
    # set seed and cuDNN environment
    torch.backends.cudnn.enabled = True

    if main_options['speed_up']['ena_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info('TF32 enabled')
    logger.warning('CuDNN for acceleration enabled by setting torch.backends.cudnn.enabled as True')
    TheUtil.set_seed(main_options['seed'])
    if main_options['use_non_deterministic']:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        logger.warning('Use non deterministic algorithm')

    # set log file (only rank0)
    if TheUtil.is_main_process():
        logger.add(main_options['path']['log_file'], level=main_options['log_level'])

    # set dataloader
    is_distributed = main_options['distributed']['enable_fsdp']
    train_loaders_dict, valid_loaders_dict = TheData.create_dataloader(main_options, is_distributed=is_distributed)

    # set trainer
    trainer = TheTrainer.create_trainer(main_options, train_loaders_dict, valid_loaders_dict)

    # modify options based on different phase
    main_options['path']['profile_path'] = None

    if main_options['phase'] == 'debug':
        main_options['train']['val_iter'] = 20
        logger.critical('DEBUG MODE is enabled, val_iter is set to 20!')
    elif main_options['phase'] == 'profile':
        main_options['path']['profile_path'] = os.path.join(main_options['path']['tensorboard_log_dir'], 'profile')
        logger.critical('PROFILE MODE is enabled.')
    elif main_options['phase'] == 'detect':
        torch.autograd.set_detect_anomaly(True)
        logger.critical('DETECT MODE is enabled.')

    try:
        trainer.train(main_options)
    finally:
        logger.info('End Execution')


if __name__ == '__main__':
    # Detect distributed environment (torchrun sets RANK, LOCAL_RANK, WORLD_SIZE)
    if "RANK" in os.environ:
        local_rank = TheUtil.setup_distributed()
        logger.info('Distributed training initialized: rank={}, local_rank={}, world_size={}'.format(
            os.environ['RANK'], local_rank, os.environ['WORLD_SIZE']))
    else:
        local_rank = 0

    logger.info('Start Execution...')
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'debug', 'profile', 'detect'], help='Run train, debug, profile or detect', default='train')

    args = parser.parse_args()

    options = TheUtil.parse_config_json(args)
    options['local_rank'] = local_rank

    try:
        main(options)
    finally:
        TheUtil.cleanup_distributed()
