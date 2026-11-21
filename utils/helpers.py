import json, random
from pathlib import Path
import numpy as np
import torch


def get_device():
    """Choose CUDA, Apple Silicon MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def synchronize_device(device):
    """Wait for asynchronous accelerator work when timing inference."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    from models.unet import UNet
    model = UNet(in_channels=4, out_channels=1, base=ckpt.get('base', 24)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')
