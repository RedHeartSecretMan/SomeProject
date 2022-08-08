# encoding:utf-8
import sys
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class AccuracyProportion(nn.Module):
    def __init__(self, proportion=0.5, threshold=None, ignore_channels=None):
        super().__init__()
        self.proportion = proportion
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, pr, gt, eps):
        assert pr.shape[0] == gt.shape[0]

        pr = _threshold(pr, threshold=self.threshold)
        pr, gt = _take_channels(pr, gt, ignore_channels=self.ignore_channels)

        sum_nums = pr.shape[0]
        nums = 0
        for idx in range(sum_nums):
            intersection = torch.sum(gt[idx] * pr[idx])
            union = torch.sum(gt[idx]) + torch.sum(pr[idx]) - intersection + eps
            score = (intersection + eps) / union

            if score > self.proportion:
                nums += 1
        ap_score = nums / sum_nums

        return ap_score


class ExpectedAverageOverlap(nn.Module):
    def __init__(self, threshold=None, ignore_channels=None):
        super().__init__()
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, pr, gt):
        assert pr.shape[0] == gt.shape[0]

        pr = _threshold(pr, threshold=self.threshold)
        pr, gt = _take_channels(pr, gt, ignore_channels=self.ignore_channels)

        nums = pr.shape[0]
        score = 0
        for idx in range(nums):
            temp = 0
            for k in range(idx):
                tp = torch.sum((gt[k] == pr[k]).type(pr.dtype))
                temp += tp / gt[k].view(-1).shape[0]

            score += (temp/idx)
        eao_score = score / nums

        return eao_score


def loss_functions(loss_name):
    if loss_name == 'dice':
        return smp.utils.metrics.Fscore(beta=1, eps=1e-7, threshold=0.5)
    elif loss_name == 'ac':
        return smp.utils.metrics.Accuracy(threshold=0.5)
    elif loss_name == 'pre':
        return smp.utils.metrics.Precision(threshold=0.5)
    elif loss_name == 'r':
        return smp.utils.metrics.Recall(threshold=0.5)
    elif loss_name == 'f1':
        return smp.utils.metrics.Fscore(beta=1, eps=1e-7)
    elif loss_name == 'iou':
        return smp.utils.metrics.IoU(eps=1e-7, threshold=0.5)
    elif loss_name == 'ap@0.5':
        return AccuracyProportion(proportion=0.5, threshold=0.5)
    elif loss_name == 'ap@0.7':
        return AccuracyProportion(proportion=0.7, threshold=0.5)
    elif loss_name == 'eao':
        return ExpectedAverageOverlap(threshold=0.5)
    else:
        print('The loss function name: {} is invalid'.format(loss_name))
        sys.exit()


if __name__ == "__main__":
    a = torch.randn(4, 1, 2, 2)
    b = torch.randn(4, 1, 2, 2)
    c = loss_functions("dice")(a, b)
    print(c)
