from pathlib import Path
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def key_for(path):
    s = path.stem.lower()
    for token in ('mask', 'masks', 'manual', 'label', 'labels', 'segmentation', 'seg'):
        s = re.sub(rf'(^|[_-]){token}($|[_-])', r'\1', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def discover_pairs(root):
    """Discover image/mask directories and return exact normalized-stem matches."""
    root = Path(root)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not files:
        raise FileNotFoundError(f'No image files found under {root}')
    mask_dirs = [p for p in root.rglob('*') if p.is_dir() and any(w in p.name.lower() for w in ('mask', 'label', 'groundtruth', 'ground_truth'))]
    # In common Kaggle layout the sibling is named masks; avoid treating CXR_png as masks.
    mask_files = [f for f in files if any(m == f.parent or m in f.parents for m in mask_dirs)]
    if not mask_files:
        mask_files = [f for f in files if re.search(r'(^|[_-])(mask|masks|manual|label|labels|seg)([_-]|$)', f.stem.lower())]
    mask_ids = {id(p) for p in mask_files}
    image_files = [f for f in files if id(f) not in mask_ids and not re.search(r'(^|[_-])(mask|masks|manual|label|labels|seg)([_-]|$)', f.stem.lower())]
    by_key = {}
    for m in mask_files:
        by_key.setdefault(key_for(m), []).append(m)
    pairs, used = [], set()
    for im in sorted(image_files):
        cand = by_key.get(key_for(im), [])
        cand = [m for m in cand if m not in used]
        if cand:
            # Prefer the mask with closest relative path/name if duplicate stems exist.
            mask = sorted(cand, key=lambda p: (p.suffix.lower() != im.suffix.lower(), len(p.parts)))[0]
            pairs.append((im, mask)); used.add(mask)
    if not pairs:
        raise RuntimeError(f'Found {len(image_files)} image candidates and {len(mask_files)} mask candidates under {root}, but no filename matches. Inspect filenames and use matching stems.')
    return pairs


class LungDataset(Dataset):
    def __init__(self, pairs, size=256):
        self.pairs = pairs
        self.size = int(size)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert('L').resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert('L').resize((self.size, self.size), Image.Resampling.NEAREST)
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(x[None]), torch.from_numpy(y[None]), str(image_path)
