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


def dataloader(img_dir):
    print(img_dir)
    # read image

    files = os.listdir(img_dir)
    file = files[random.randint(1,50)]

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

    # setup data_loader instances
    file = os.path.join(config['data_loader']['args']['data_dir'],
                        os.listdir(config['data_loader']['args']['data_dir'])[0])
    # data_loader = TomatoPredDataLoader(path=file,batch_size=2,shuffle=False)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu') if not torch.cuda.is_available() else None)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    start_time = datetime.datetime.now()

    datas = dataloader(config['data_loader']['args']['data_dir'])

    outs = []
    res = torch.zeros(datas[0].shape[1], datas[0].shape[2])
    resGpu = res.to(device)
    with torch.no_grad():
        for i, data in enumerate(tqdm(datas)):
            data = data.reshape(1, data.shape[0], data.shape[1], data.shape[2])
            data = data.to(device)
            output = model(data)

            outs.append(output)

    # displayPredictions(outs)
    outs = arrToCpu(outs)
    ogs = inverseAugmentation(outs)

    for i in range(len(ogs)):
        assert (ogs[0].shape == ogs[i].shape)

    ogs = arrToGpu(ogs, device)

    for o in ogs:
        o = o.squeeze()
        threshold = torch.where(o > config["cutoff"], 1,0)
        resGpu.add_(threshold)

    resGpu = torch.where(resGpu > config["intersectionCount"],1,0)

    # displayPredictions(ogs)

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
    save_image("new_output", original_input.numpy(), intersection.numpy())

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000

    print(execution_time)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(intersection, cmap=cm.jet)
    ax[0].set_title('Intersection')

    ax[1].imshow(intersection)
    ax[1].set_title('Target mask')

    ax[2].imshow(ogs[3])
    ax[2].set_title('Pred mask')

    ax[3].imshow(original_input)
    ax[3].set_title('Original')

    plt.tight_layout()
    plt.show()

    # print(result)
    n_samples = len(datas)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

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
    # dataloader(config["data_loader"]["args"]["label_dir"])
