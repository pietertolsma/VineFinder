import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torchvision import transforms

from matplotlib import cm
import matplotlib.pyplot as plt


def custom_loss(pred, target):
    iflat = pred.view(-1)
    tflat = target.view(-1)
    return 1 - (iflat - iflat * tflat).sum() / (iflat.sum() + 1e-6)


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=3,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

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

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            a = output
            b = target
            d = data
            try:
                output = output.cpu()
            except:
                output = a

            try:
                target = target.cpu()
            except:
                target = b

            try:
                data = data.cpu()
            except:
                data = d

            print(output.shape)
            for i in range(output.shape[0]):
                img = torch.nn.Sigmoid()(output[i])[i,0,:,:]
                # img = output[i, 0, :, :]

                maskpred = (img > config["cutoff"]) * 255

                fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
                ax = axes.ravel()

                ax[0].imshow(img * (img > config['cutoff']), cmap=cm.jet)
                ax[0].set_title('Prediction')

                ax[1].imshow(target[i, 0, :, :])
                ax[1].set_title('Target mask')

                ax[2].imshow(maskpred)
                ax[2].set_title('Pred mask')

                original_input = data[i]
                original_input = original_input.transpose(0, 2)
                original_input = original_input.transpose_(0, 1)
                ax[3].imshow(original_input)
                ax[3].set_title('Original')

                plt.tight_layout()
                plt.show()

                yn = input()
                if(yn == "y"):
                    print("should be added to the trainset")

                # #plt.savefig("output/" + file.split("/")[-1])
                # plt.close()

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


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