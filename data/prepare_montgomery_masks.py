"""Fuse the official left/right Montgomery manual masks for binary lung training."""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent / "MontgomerySet"
IMAGES = ROOT / "CXR_png"
LEFT = ROOT / "ManualMask" / "leftMask"
RIGHT = ROOT / "ManualMask" / "rightMask"
OUT = IMAGES / "FusedMasks"
OUT.mkdir(parents=True, exist_ok=True)

image_paths = sorted(IMAGES.glob("*.png"))
if len(image_paths) != 138:
    raise RuntimeError(f"Expected 138 official radiographs, found {len(image_paths)}")

for image_path in image_paths:
    left_path, right_path = LEFT / image_path.name, RIGHT / image_path.name
    if not left_path.is_file() or not right_path.is_file():
        raise FileNotFoundError(f"Missing paired lung masks for {image_path.name}")
    left = np.asarray(Image.open(left_path).convert("L")) > 0
    right = np.asarray(Image.open(right_path).convert("L")) > 0
    if left.shape != right.shape:
        raise ValueError(f"Mask dimension mismatch for {image_path.name}: {left.shape} / {right.shape}")
    fused = np.logical_or(left, right).astype(np.uint8) * 255
    Image.fromarray(fused).save(OUT / image_path.name)

print(f"Prepared {len(image_paths)} fused binary lung masks in {OUT}")
