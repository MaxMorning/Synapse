import random

from torch.utils.data import DataLoader, Subset
from util.util import init_obj, set_seed
from loguru import logger
from torch.utils.data.dataloader import Sampler
from functools import partial
import torch


def create_datasets(options):
    """ loading Dataset() class from given file's name """
    train_dataset_opt = options['datasets']['which_dataset']['train_dataset']
    valid_dataset_opt = options['datasets']['which_dataset']['valid_datasets']

    if train_dataset_opt is not None:
        train_dataset = init_obj(train_dataset_opt, default_file_name='data.dataset', init_type='Dataset')
        logger.info("Train dataset has {} samples.".format(len(train_dataset)))
    else:
        train_dataset = None

    if valid_dataset_opt is not None:
        val_datasets_dict = {}
        for name, valid_dataset_opt in valid_dataset_opt.items():
            val_datasets_dict[name] = init_obj(valid_dataset_opt, default_file_name='data.dataset', init_type='Dataset')
            logger.info("Valid dataset {} has {} samples.".format(name, len(val_datasets_dict[name])))
    else:
        val_datasets_dict = None

    return train_dataset, val_datasets_dict


def create_dataloader(options):
    train_dataloader_args = options['datasets']['dataloader']['train_args']
    val_dataloader_args = options['datasets']['dataloader']['val_args']

    generator = torch.Generator()
    generator.manual_seed(options['seed'])

    train_dataset, val_datasets_dict = create_datasets(options)
    worker_init_fn = partial(set_seed, gl_seed=options['seed'])

    ''' create dataloader and validation dataloader '''
    if train_dataset is not None:
        # paired_sampler = ReproducibleSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, worker_init_fn=worker_init_fn, generator=generator, **train_dataloader_args)
        logger.success('Train Loader Created.')
    else:
        train_dataloader = None

    val_dataloaders_dict = {}
    for val_dataset_name, val_dataset_inst in val_datasets_dict.items():
        # val_sampler = ReproducibleSampler(val_dataset_inst)
        val_dataloaders_dict[val_dataset_name] = DataLoader(val_dataset_inst, worker_init_fn=worker_init_fn, generator=generator,
                                                            **val_dataloader_args)
    logger.success('Valid Loader Created.')
    return train_dataloader, val_dataloaders_dict


class ReproducibleSampler(Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return ReproducibleIter(self.__len__())

    def __len__(self):
        return len(self.data_source)


class ReproducibleIter:
    def __init__(self, data_count):
        self.data_count = data_count
        self.current_idx = 0
        self.shuffle_list = list(range(self.data_count))
        random.shuffle(self.shuffle_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx < self.data_count:
            element = self.shuffle_list[self.current_idx]
            self.current_idx += 1
            return element
        else:
            raise StopIteration

class InfinityIterator:
    def __init__(self, collection):
        assert len(collection) > 0

        self.collection = collection
        self.iter = iter(collection)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.collection)
            return next(self.iter)

    def __len__(self):
        return len(self.collection)
