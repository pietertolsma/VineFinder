import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
import cv2
import torch

def save_image(folder, original_image, mask, filename):
    """
    Saves the original image and the mask to a folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder + "/images")
        os.makedirs(folder + "/masks")

    # files = os.listdir(f'{folder}/images')
    # file_count = len(files)

    image_name = f'images/{filename}.png'
    mask_name = f'masks/mask_{filename}.png'
    original_image_path = os.path.join(folder, image_name)
    mask_path = os.path.join(folder, mask_name)

    plt.imsave(original_image_path, original_image)
    cv2.imwrite(mask_path, mask)
    #plt.imsave(mask_path, mask)


# test_image = plt.imread('../data/grabpoints/test/images/image_0_0.png')
# test_mask = torch.Tensor(np.random.randint(0, 1, (test_image.shape[0], test_image.shape[1])))
# print(test_mask)

# save_image("test_output", test_image, test_mask)