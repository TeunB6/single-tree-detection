"""
Compute channel-wise mean and stdev for normalization transform
"""

from pathlib import Path
import numpy as np
import torch
from typing import Iterable


def get_images(data_dir: Path | Iterable[Path]):
    files = []
    if isinstance(data_dir, Path):
        files.extend(sorted(data_dir.glob("*.pt")))
    elif isinstance(data_dir, Iterable):
        files = []
        for dir in data_dir:
            files.extend(sorted(dir.glob("*.pt")))

    unpickles = [torch.load(file, weights_only=False) for file in files]
    images = [np.asarray(image) for image, boxes in unpickles]
    images = np.stack(images)
    return images


def means(images):
    np.mean(images, axis=-1)


if __name__ == "__main__":
    images = get_images(Path("data/neon_tree/NeonTreeEvaluation/training/RGB"))

    print("Means:               ", means(images))
