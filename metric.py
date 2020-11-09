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

