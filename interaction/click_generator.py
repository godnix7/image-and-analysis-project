import random
import torch
import torch.nn.functional as F


def sample_error_click(pred, target):
    """Sample a click in the largest false-negative/false-positive component."""
    fn = (target > .5) & (pred < .5)
    fp = (target <= .5) & (pred >= .5)
    # Favor the larger error region, with a random tie-break.
    areas = [int(fn.sum()), int(fp.sum())]
    if max(areas) == 0: return None
    positive = areas[0] > areas[1] or (areas[0] == areas[1] and random.random() < .5)
    ys, xs = torch.where(fn[0, 0] if positive else fp[0, 0])
    j = random.randrange(len(xs))
    return (int(xs[j]), int(ys[j]), positive)


def add_click(pos, neg, click, radius=12):
    if click is None: return pos, neg
    x, y, positive = click
    h, w = pos.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=pos.device), torch.arange(w, device=pos.device), indexing='ij')
    g = torch.exp(-((xx-x)**2 + (yy-y)**2) / (2 * radius**2))[None, None]
    if positive: pos = torch.maximum(pos, g)
    else: neg = torch.maximum(neg, g)
    return pos, neg
