import numpy as np


def binary_metrics(pred, target, eps=1e-8):
    p = np.asarray(pred).astype(bool); t = np.asarray(target).astype(bool)
    tp = np.logical_and(p, t).sum(); fp = np.logical_and(p, ~t).sum(); fn = np.logical_and(~p, t).sum()
    dice = (2*tp + eps)/(2*tp + fp + fn + eps)
    iou = (tp + eps)/(tp + fp + fn + eps)
    precision = (tp + eps)/(tp + fp + eps)
    recall = (tp + eps)/(tp + fn + eps)
    return {'dice': float(dice), 'iou': float(iou), 'precision': float(precision), 'recall': float(recall), 'f1': float(2*precision*recall/(precision+recall+eps))}
