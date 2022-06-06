from math import floor
import torch
import random
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torchvision.io import read_image
from base import BaseDataLoader

from torch.utils.data import Dataset

import os, os.path
from PIL import Image
import numpy as np

from skimage.color import rgb2hsv, rgb2gray
from skimage.transform import (probabilistic_hough_line)
from skimage.feature import canny

class PadImage(object):
    """
    """

    def __call__(self, pic, size=(572, 572)):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        pic = transforms.Resize(size=(min(pic.shape[1], size[0]), min(pic.shape[2], size[1])))(pic)
        current_width, current_height = pic.shape[2], pic.shape[1]
        add_width = size[0] - current_width
        add_height = size[1] - current_height
        padding = (floor(add_width / 2), floor(add_height / 2), add_width - floor(add_width / 2),
                   add_height - floor(add_height / 2))
        padded = F.pad(pic, padding, 0, 'constant').float()
        return padded

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TomatoImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.img_dir = root_dir + "/unlabled"
        self.img_files = os.listdir(self.img_dir)
        print(self.img_files)
        self.mask_dir = root_dir + "/masks"
        self.mask_files = os.listdir(self.mask_dir)
        self.mask_files.sort()
        print(self.mask_files)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = read_image(img_path)
        mask = read_image(mask_path)

        # Ensure that both random operations are the same

        rotation = random.randint(-90, 90)
        flip = random.uniform(0, 1) > 0.5

        if flip:
            transforms.functional.hflip(image)
            transforms.functional.hflip(mask)

        transforms.functional.rotate(mask, rotation)
        transforms.functional.rotate(image, rotation)

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        #  image[0, :, :] = canny(image[0, :, :], sigma=4)
        # standardize image between 0, 1
        image = image / 255.0
        # image = image * (image < 5.6)

        # import matplotlib.pyplot as plt

        # #tensor_image.permute(1, 2, 0)
        # plt.imshow(image.permute(1, 2, 0).float() )
        # plt.show()

        # mask = mask[0, :, :]
        # plt.imshow(mask)
        # plt.show()

        return image, mask


class TomatoDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            PadImage(),
        ])
        m_trsfm = transforms.Compose([
            PadImage(),
        ])

        self.data_dir = data_dir
        self.dataset = TomatoImageDataset(self.data_dir, transform=trsfm, mask_transform=m_trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
