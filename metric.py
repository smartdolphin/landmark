import pandas as pd
import numpy as np
import torch


def GAP(pred: torch.Tensor, target: torch.Tensor) -> float:
    ''' Simplified GAP@1 metric: only one prediction per sample is supported '''
    confs, predicts = torch.max(pred.detach(), dim=1)
    _, indices = torch.sort(confs, descending=True)

    predicts = predicts[indices]
    target = target[indices]

    res, true_pos = 0.0, 0

    for i, (p, t) in enumerate(zip(predicts, target)):
        rel = int(p == t)
        true_pos += rel

        res += true_pos / (i + 1) * rel

    res /= target.shape[0] # FIXME: incorrect, not all test images depict landmarks
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.reset()
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(pred, label):
    num_correct = torch.sum(pred.max(1)[1] == label.data)
    num_cnt = len(label)
    acc = (num_correct.double()/num_cnt).cpu() * 100
    return acc

