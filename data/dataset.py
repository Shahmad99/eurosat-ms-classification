"""
EuroSAT GeoTIFF dataset loader.

Modalities
----------
"rgb"  — 3 channels: B04 (red), B03 (green), B02 (blue), read directly from
         file order.  Suitable for the 3-channel RGB-MoCo backbone.

"ms"   — 10 channels: all land bands after reordering to SSL4EO-S12 canonical
         order.  Specifically:
           1. Reorder 13 file-order bands → canonical order (corrects B8A placement)
           2. Drop atmospheric bands B01, B09, B10 (canonical positions 0, 9, 10)
           → Kept bands in order: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
         Suitable for the 10-channel MS backbone (conv1 adapted from 13-ch pretrained).

Band index references come from config.yaml so they can be changed in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config


class EuroSATDataset(Dataset):
    """Loads EuroSAT Sentinel-2 GeoTIFFs and returns (tensor, label) pairs.

    Args:
        paths:     paths to .tif patch files
        labels:    integer class label for each path
        modality:  "rgb" → (3, H, W) | "ms" → (10, H, W)
        transform: callable applied to the float32 tensor before returning
    """

    def __init__(
        self,
        paths: Sequence[str],
        labels: Sequence[int],
        modality: str = "ms",
        transform=None,
    ) -> None:
        assert len(paths) == len(labels)
        assert modality in ("rgb", "ms"), f"Unknown modality '{modality}'"
        self.paths     = list(paths)
        self.labels    = list(labels)
        self.modality  = modality
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        with rasterio.open(self.paths[idx]) as src:
            data = src.read()                                    # (13, H, W) uint16

        tensor = torch.from_numpy(data.astype(np.float32))      # (13, H, W)

        if self.modality == "rgb":
            # Select R/G/B directly from file-order indices — no reorder needed
            tensor = tensor[config.RGB_FILE_IDX]                 # (3, H, W)
        else:
            # Reorder to canonical SSL4EO-S12 order, then keep the 10 land bands
            tensor = tensor[config.MS_REORDER]                   # (13, H, W) canonical
            tensor = tensor[config.KEPT_MS_IDX]                  # (10, H, W) no atmos

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, self.labels[idx]


def make_loader(
    paths: list[str],
    labels: list[int],
    modality: str,
    transform,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    ds = EuroSATDataset(paths, labels, modality=modality, transform=transform)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = config.NUM_WORKERS,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = drop_last,
    )
