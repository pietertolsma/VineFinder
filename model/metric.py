import torch
from .loss import dice_loss

def accuracy(output, target):
    return dice_loss(output, target)


def top_k_acc(output, target, k=3):
    return 0
    # with torch.no_grad():
    #     pred = torch.topk(output, k, dim=1)[1]
    #     assert pred.shape[0] == len(target)
    #     correct = 0
    #     for i in range(k):
    #         correct += torch.sum(pred[:, i] == target).item()
    # return correct / len(target)
