import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

folder = "new_output"

original_images = os.listdir(f'{folder}/images')
masks = os.listdir(f'{folder}/masks')

original_images.sort()
masks.sort()

image_count = len(original_images)
width = math.ceil(math.sqrt(image_count))

fig = plt.figure(figsize=(width, width))

i = 0
for img_file in original_images:
    mask_file = masks[i]

    original_img = plt.imread(f'{folder}/images/{img_file}')

    mask_img = cv2.imread(f'{folder}/masks/{mask_file}')

    original_img[:, :, 1] = original_img[:, :, 1] + 0.5 * mask_img[:, :, 0]

    masked = np.ma.masked_where(mask_img == 0, original_img[:, :, 0:3])

    fig.add_subplot(width, width, i + 1)

    plt.imshow(original_img, interpolation="none")

    i += 1

plt.show()