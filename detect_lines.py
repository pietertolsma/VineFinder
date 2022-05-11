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
                if abs(cluster[0] - angle) < np.pi/6:
                    cluster[1].append(line)
                    found = True
                    break
            if not found:
                clusters.append([angle, [line]])

    clusters.sort(key=lambda x: len(x[1]), reverse=True)
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

def save_plot(file, image, tracks, edges):
    # Generating figure 2

    # clusters = group_lines(lines)
    # print(len(clusters))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(imread(file), cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(image, cmap=cm.gray)
    ax[1].set_title('Isolated Vines')

    ax[2].imshow(edges * 0)
    i=0
    # for (angle, cluster) in clusters:
    #     for line in cluster:
    #         p0, p1 = line
    #         color = 'r' if i == 0 else 'b'
    #         ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), color=color)
    #     i += 1

    i = 0

    color = iter(cm.rainbow(np.linspace(0, 1, len(tracks))))

    sizes = list(map(lambda x: track_length(x), tracks))
    print(str(sizes))

    for track in tracks:
        thiscolor = next(color)
        for line in track:
            p0, p1 = line
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]), color=thiscolor)
        i += 1

    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
        #a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()
    #plt.savefig("output/" + file.split("/")[-1])
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

    tracks = all_tracks(lines,20,0.2)
    avg = 0
    avgCount = 0
    for t in tracks:
        avg += track_length(t)
        avgCount += len(t)

    avg /= len(tracks)
    avgCount /= len(tracks)

    reduced_tracks = [t for t in tracks if track_length(t) > 2*avg and len(t) > avgCount]

    print("avg len(track)=={}".format(avg))
    save_plot(img, image, reduced_tracks, edges)
    
    return lines

def point_distance(a,b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def lines_angle(a,b):
    aa = get_angle(a)
    ab = get_angle(b)

    return max(aa-ab,ab-aa)

def line_distance(a, b):
    p0, p1 = a
    p2, p3 = b

    return min(point_distance(p0,p2), point_distance(p0,p3), point_distance(p1,p2), point_distance(p1,p3))

#A function that calculates the total length of all lines in the list
def total_length(lines):
    total = 0
    for line in lines:
        p0, p1 = line
        total += point_distance(p0,p1)
    return total

# A function that connects lines that are near each other and have the same angle.
def all_tracks(lines, min_length, min_distance):
    tracks = []
    while len(lines) > 0:
        line = lines.pop()
        track = [line]
        while True:
            closest_line = find_closest_line(track[-1], lines)
            if closest_line is None:
                break
            if lines_angle(track[-1], closest_line) > np.pi/6:
                break
            if line_distance(track[-1], closest_line) > min_distance:
                break
            track.append(closest_line)
            lines.remove(closest_line)
        if total_length(track) > min_length:
            tracks.append(track)
    return tracks

def connect_lines(current, used, lines, lengthCutoff, angleCutoff):
    lowestAngle = angleCutoff
    lowestLine = []
    front = -1
    for line in lines:
        if not current:
            current = [line]
            continue

        if line_distance(current[0],line) < lengthCutoff and lines_angle(current[0],line) < lowestAngle and line not in used:
            # current.insert(0,line)
            lowestAngle = lines_angle(current[0],line)
            lowestLine = line
            front = 0

        if line_distance(current[-1],line) < lengthCutoff and lines_angle(current[-1],line) < lowestAngle and line not in used:
            # current.append(line)
            lowestAngle = lines_angle(current[-1], line)
            lowestLine = line
            front = 1

    if front == -1:
        return current
    elif front == 0:
        current.insert(0, lowestLine)
    elif front == 1:
        current.append(lowestLine)

    used.append(lowestLine)
    return connect_lines(current,used,lines,lengthCutoff, angleCutoff)

def line_length(line):
    p0, p1 = line
    return point_distance(p0, p1)

def track_length(lines):
    len = 0
    for line in lines:
        len += line_length(line)

    return round(len,2)

def all_tracks(lines, lengthCutoff = 0.1, angleCutoff = 0.1):
    tracks = []
    for line in lines:
        tracks.append(connect_lines([line],[line],lines,lengthCutoff, angleCutoff))
    return tracks
