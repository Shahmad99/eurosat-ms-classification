"""
Image transforms for EuroSAT Sentinel-2 patches.

SentinelTransform  — preprocessing for both the RGB-MoCo and MS-MoCo backbones:
                     divide by 10 000 to get reflectance, resize to 256,
                     centre-crop to 224.  Matches the SSL4EO-S12 pretraining pipeline.

AugmentTransform   — wraps any base transform with random hflip / vflip / 90° rotation.
                     Land-cover classification is rotation-invariant, so all four
                     rotations are valid augmentations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class SentinelTransform:
    """DN → reflectance, resize to 256×256, centre-crop to 224×224."""

    def __call__(self, x: Tensor) -> Tensor:
        x = F.interpolate(
            x.unsqueeze(0), size=(256, 256),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        h, w = x.shape[-2], x.shape[-1]
        t = (h - 224) // 2
        l = (w - 224) // 2
        x = x[:, t : t + 224, l : l + 224]
        return x / 10_000.0


class AugmentTransform:
    """Random hflip, vflip, and 90° rotation applied after the base transform."""

    def __init__(self, base) -> None:
        self.base = base

    def __call__(self, x: Tensor) -> Tensor:
        x = self.base(x)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, [-1])
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, [-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, [-2, -1])
        return x


def get_transform(augment: bool = False):
    """Return SentinelTransform, optionally wrapped with augmentation."""
    base = SentinelTransform()
    return AugmentTransform(base) if augment else base
