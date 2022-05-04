import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray
from skimage.transform import (probabilistic_hough_line)
from skimage.feature import canny

import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd

def mask_vine(image):
    hsv = rgb2hsv(image)
    saturation_mask = hsv[:,:,1] > 0.34
    hue_mask = (hsv[:,:,0] > 0.080) * (hsv[:,:, 0] < 0.360)
    return rgb2gray(image) * saturation_mask * hue_mask

def save_plot(file, image, lines, edges):
    # Generating figure 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(imread(file), cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(image, cmap=cm.gray)
    ax[1].set_title('Isolated Vines')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
        #a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.savefig("output/" + file.split("/")[-1])
    plt.close()


def detect_lines(img):
    """
    Detects lines in an image.
    """

    image = imread(img)
    image = mask_vine(image)
    edges = canny(image, sigma=4)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                    line_gap=3)

    save_plot(img, image, lines, edges)
    
    return lines