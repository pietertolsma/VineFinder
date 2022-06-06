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


class TomatoPredImageDataset(Dataset):
    def __init__(self, path):
        self.images = self.transforms(path)

    def transforms(self, filePath):
        # read image
        print("file: " + str(filePath))
        file = read_image(filePath)
        # print("file: " + file)

        # img = Image.open(img_dir + "/" + file)
        # print(img)
        # transform = transforms.Compose([
        #     transforms.PILToTensor(),
        #     PadImage()
        # ])

        # img = transform(img)
        imgs = [file]
        # rotate 90,180 and 270 deg
        for i in range(3):
            imma = F.rotate(imgs[0], 90 * i)
            imgs.append(imma)

        assert (len(imgs) == 4)
        trans = []
        # for each of those
        for img in imgs:
            trans.append(img)
            # flip vertically
            trans.append(transforms.functional.hflip(img))
            # flip horizontally
            trans.append(transforms.functional.vflip(img))

        trsfm = transforms.Compose([
            PadImage(),
        ])

        out = []
        for i in trans:
            out.append(trsfm(i))

        return trans

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class TomatoPredDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, path, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.path = path
        print("file: " + str(self.path))
        self.dataset = TomatoPredImageDataset(self.path)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
