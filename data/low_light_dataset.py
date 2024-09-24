import torch.utils.data as data
import os
import random
import torch.nn.functional as F

from PIL import Image
from util.util import tensor2img
from torchvision import transforms
import torch
import numpy as np
from torchvision.io import read_image


from util.util import set_seed

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.npy', '.webp'
]

class LOLv2PairedDataset(data.Dataset):
    def __init__(self, paired_path, image_size):
        self.paired_files = list_image_path(os.path.join(paired_path, 'low'),
                                            os.path.join(paired_path, 'normal'))

        self.all_files = self.paired_files

        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        low_image = Image.open(self.all_files[index][0]).convert('RGB')

        file_name = os.path.basename(self.all_files[index][0])

        if self.all_files[index][1] is None:
            low_image = self.transform(low_image)
            return file_name, low_image, low_image
        else:
            high_image = Image.open(self.all_files[index][1]).convert('RGB')

            torch_rng = torch.random.get_rng_state()
            low_image = self.transform(low_image)

            torch.random.set_rng_state(torch_rng)
            high_image = self.transform(high_image)
            return {
                "file_name": file_name,
                "input": low_image,
                "ground_truth": high_image
            }

    def __len__(self):
        return len(self.all_files)


def list_image_path(low_light_path, high_light_path):
    files = os.listdir(low_light_path)
    files.sort()
    image_files = []
    for file in files:
        if os.path.splitext(file)[-1] in IMG_EXTENSIONS:
            image_files.append([os.path.join(low_light_path, file),
                                os.path.join(high_light_path, file) if high_light_path is not None else None])

    return image_files


class PngEvalLoader(data.Dataset):
    def __init__(self, paired_path, image_list):
        all_paired_files = list_image_path(os.path.join(paired_path, 'low/teacher'),
                                           os.path.join(paired_path, 'normal'))

        self.paired_files, self.image_count = split_select_file(all_paired_files, image_list)
        
        self.full_set = False

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.image_size = None

    def __getitem__(self, index):
        low_image = load_image(self.paired_files[index][0], self.transform)
        file_name = os.path.basename(self.paired_files[index][0])

        high_image = load_image(self.paired_files[index][1], self.transform)

        return {
            "file_name": file_name,
            "input": low_image,
            "ground_truth": high_image
        }

    def __len__(self):
        if self.full_set:
            return len(self.paired_files)
        else:
            return self.image_count


def load_image(path, transform):
    image = Image.open(path).convert('RGB')
    image = transform(image)

    return image


def split_select_file(all_paired_files, image_list):
    if image_list is None:
        paired_files = all_paired_files
        image_count = len(all_paired_files)

    else:
        selected_paired_files = []
        unselected_paired_files = []

        for image_pair in all_paired_files:
            if os.path.basename(image_pair[0]) in image_list:
                selected_paired_files.append(image_pair)
            else:
                unselected_paired_files.append(image_pair)

        paired_files = selected_paired_files + unselected_paired_files

        image_count = len(selected_paired_files)

    return paired_files, image_count
