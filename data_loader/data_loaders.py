
from math import floor
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torchvision.io import read_image
from base import BaseDataLoader

from torch.utils.data import Dataset

import os, os.path
from PIL import Image
import numpy as np

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
        padding = (floor(add_width / 2), floor(add_height / 2), add_width - floor(add_width / 2), add_height - floor(add_height / 2))
        padded = F.pad(pic, padding, 0, 'constant')
        return padded
        

    def __repr__(self):
        return self.__class__.__name__ + '()'

class TomatoImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.img_dir = root_dir + "/images"
        self.img_files = os.listdir(self.img_dir)
        self.mask_dir = root_dir + "/masks"
        self.mask_files = os.listdir(self.mask_dir)
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = read_image(img_path)
        mask = read_image(mask_path)
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

class TomatoDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    # def preprocess(pil_img, scale, is_mask):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    #     pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    #     img_ndarray = np.asarray(pil_img)

    #     if not is_mask:
    #         if img_ndarray.ndim == 2:
    #             img_ndarray = img_ndarray[np.newaxis, ...]
    #         else:
    #             img_ndarray = img_ndarray.transpose((2, 0, 1))

    #         img_ndarray = img_ndarray / 255

    #     return img_ndarray

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.Grayscale(),
            PadImage()
        ])
        m_trsfm = transforms.Compose([
            PadImage()
        ])
        self.data_dir = data_dir
        self.dataset = TomatoImageDataset(self.data_dir, transform=trsfm, mask_transform=m_trsfm)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
