import argparse
import datetime
import os
import random
import time

import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader.labeler_loader import PadImage, TomatoPredDataLoader
from parse_config import ConfigParser
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

from utils.data_saver import save_image


def custom_loss(pred, target):
    iflat = pred.view(-1)
    tflat = target.view(-1)
    return 1 - (iflat - iflat * tflat).sum() / (iflat.sum() + 1e-6)


def dataloader(index, img_dir):
    print(img_dir)
    # read image

    files = os.listdir(img_dir)
    file = files[index]

    print("file: " + file)

    img = Image.open(img_dir + "/" + file)
    print(img)
    transform = transforms.Compose([
        transforms.PILToTensor(),
        PadImage()
    ])

    img = transform(img)/255
    imgs = [img]
    # rotate 90,180 and 270 deg
    for i in range(1,4):
        imma = F.rotate(imgs[0], 90 * i)
        imgs.append(imma)

    assert (len(imgs) == 4)
    out = []
    # for each of those
    for img in imgs:
        out.append(img)
        # flip vertically
        out.append(transforms.functional.hflip(img))
        # flip horizontally
        out.append(transforms.functional.vflip(img))
    # displayInputs(out)
    return out

def inverseAugmentation(datas):
    out = []
    for i in range(0, len(datas), 3):
        out.append(F.rotate(datas[i], -i * 30).squeeze())  # looks strange but at i=0 give 0deg, i=3 give 90deg etc
        out.append(F.rotate(F.hflip(datas[i+1]), -i * 30).squeeze())
        out.append(F.rotate(F.vflip(datas[i+2]), -i * 30).squeeze())

    return out


# TODO: Doesn't work for reasons unkown to mankind
def visualizeRotations(imgs):
    cols = 6
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(150, 150), sharex=True, sharey=True)

    for a in range(rows):
        for b in range(cols):
            axes[a, b].imshow(imgs[b * cols + a][0, :, :], cmap=cm.jet)

    plt.tight_layout()
    plt.show()
    # plt.savefig("output/" + file.split("/")[-1])
    plt.close()


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    start_time = datetime.datetime.now()
    condition = True
    i = 0
    while(condition and i < config["numPredictionAdditions"]):
        predImage(i, model,device)
        i+=1

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000

    print(execution_time)

def predImage(i, model, device):
    datas = dataloader(i, config['data_loader']['args']['data_dir'])

    outs = []
    res = torch.zeros(datas[0].shape[1], datas[0].shape[2])
    resGpu = res.to(device)
    with torch.no_grad():
        for i, data in enumerate(tqdm(datas)):
            data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
            data = data.to(device)
            output = model(data)

            outs.append(output)

    outs = arrToCpu(outs)
    ogs = inverseAugmentation(outs)
    ogs = arrToGpu(ogs, device)

    for o in ogs:
        o = o.squeeze()
        threshold = torch.where(o > config["cutoff"], 1,0)
        resGpu.add_(threshold)

    resGpu = torch.where(resGpu > config["intersectionCount"],1,0)

    intersection = resGpu
    i = resGpu
    try:
        intersection = resGpu.cpu()
    except:
        intersection = i

    datas = arrToCpu(datas)
    ogs = arrToCpu(ogs)

    original_input = datas[0]
    original_input = original_input.transpose(0, 2)
    original_input = original_input.transpose_(0, 1)
    save_image("new_output", original_input.numpy(), intersection.numpy(), f'image_{i}')


def arrToCpu(datas):
    out = []
    for d in datas:
        a = d
        try:
            d = d.cpu()
        except:
            d = a

        out.append(d)
    return out

def arrToGpu(datas, device):
    out = []
    for d in datas:
        d = d.to(device)

        out.append(d)
    return out

def displayPredictions(datas):
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), sharex=True, sharey=True)

    for a in range(rows):
        for b in range(cols):
            if a * cols + b < len(datas):
                axes[a, b].imshow(datas[a * cols + b].squeeze(), cmap=cm.jet)

    plt.tight_layout()
    plt.show()
    plt.close()

def displayInputs(datas):
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), sharex=True, sharey=True)

    for a in range(rows):
        for b in range(cols):
            if a * cols + b < len(datas):
                original_input = datas[a * cols + b]
                original_input = original_input.transpose(0, 2)
                original_input = original_input.transpose_(0, 1)
                axes[a, b].imshow(original_input)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
