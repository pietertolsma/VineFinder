import os
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
import numpy as np
import torch

def save_image(folder, original_image, mask):
    """
    Saves the original image and the mask to a folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder + "/images")
        os.makedirs(folder + "/masks")

    files = os.listdir(f'{folder}/images')
    file_count = len(files)

    # original_image = original_image.transpose_(0, 2)
    # original_image = original_image.transpose_(0, 1)
    mask = transforms.ToPILImage(mode='L')(mask)
    # mask = mask.transpose(0, 2)
    # mask = mask.transpose_(0, 1)

    image_name = f'images/original_image_{file_count}.png'
    mask_name = f'masks/mask_{file_count}.png'
    original_image_path = os.path.join(folder, image_name)
    mask_path = os.path.join(folder, mask_name)

    plt.imsave(original_image_path, original_image)
    plt.imsave(mask_path, mask)


# test_image = plt.imread('../data/grabpoints/test/images/image_0_0.png')
# test_mask = torch.Tensor(np.random.randint(0, 1, (test_image.shape[0], test_image.shape[1])))
# print(test_mask)

# save_image("test_output", test_image, test_mask)