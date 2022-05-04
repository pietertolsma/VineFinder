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

def gaussian_mask(size, sigma):
    """
    Generates a gaussian mask.
    """
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    y = x[:, np.newaxis]
    return np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))

def find_angles(lines):
    """
    Finds the angle of every line segment.
    """
    angles = []
    for line in lines:
        p0, p1 = line
        angles.append(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

def get_angle(line):
    """
    Finds the angle of a line.
    """
    p0, p1 = line
    return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

def group_lines(lines):
    """
    Groups lines into clusters of size n, where each cluster has a similar angle and has adjacent lines.
    """
    clusters = []
    for line in lines:
        angle = get_angle(line)
        if len(clusters) == 0:
            clusters.append([angle, [line]])
        else:
            found = False
            for cluster in clusters:
                if abs(cluster[0] - angle) < 0.1:
                    cluster[1].append(line)
                    found = True
                    break
            if not found:
                clusters.append([angle, [line]])
    return clusters

def find_closest_line(line, lines):
    """
    Finds the closest line to a given line.
    """
    closest_line = None
    closest_distance = None
    for l in lines:
        p0, p1 = l
        distance = np.sqrt((p0[0] - line[0][0])**2 + (p0[1] - line[0][1])**2)
        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_line = l
    return closest_line

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