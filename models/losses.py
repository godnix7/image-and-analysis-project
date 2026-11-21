import torch
import torch.nn.functional as F


def dice_score(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    return (2 * (pred * target).sum(dims) + eps) / (pred.sum(dims) + target.sum(dims) + eps)


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    dice = (2 * (probs * target).sum(dims) + 1e-6) / (probs.sum(dims) + target.sum(dims) + 1e-6)
    return bce + (1 - dice).mean()
