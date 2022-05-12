# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:41:37 2022

@author: Tibbe Lukkassen
"""

import os

import torch
from skimage.io import imread
from skimage.color import rgb2hsv, rgb2gray
import torchvision as tv

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import cm


def fetch_files(path):
    """
    Fetch all files in a directory
    """
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-4:] == ".png":
                files.append(os.path.join(r, file))
    return files

def mask_vine(image):
    hsv = rgb2hsv(image)
    saturation_mask = hsv[:,:,1] > 0.34
    hue_mask = (hsv[:,:,0] > 0.080) * (hsv[:,:, 0] < 0.360)
    return rgb2gray(image) * saturation_mask * hue_mask

def pad(img, size_max=256):
    """
    Pads images to the specified size (height x width). 
    """
    width, height = img.size 
    pad_height = max(0, size_max - height)
    pad_width = max(0, size_max - width)
        
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return tv.transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), (0,0,0,256))

def transform_im(img):
    image = mask_vine(img)
    image = cm.gray(image)
    
    image = torch.from_numpy(image)
    image = image.T
    
    transform = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize(255,max_size=256),
        tv.transforms.Lambda(pad)
    ])    
    image=transform(image)
    return image

files = fetch_files("./data")
    
index = 0
for file in files:
    image = imread(file)
    image = transform_im(image)
    print(image.size)
    plt.imshow(image)
    plt.show()
    if index > 0 and index % 10 == 0:
        print(f"Processed {index} files out of {len(files)}")
    index += 1
