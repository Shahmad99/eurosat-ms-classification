"""
Pretrained backbone loader.

Returns a ResNet-18 backbone with the FC head removed, ready for fine-tuning.
Output: (batch, 512) feature vectors.

Modalities
----------
"rgb"  — SENTINEL2_RGB_MOCO pretrained (3-channel input).
         Backbone is used as-is.

"ms"   — starts from SENTINEL2_ALL_MOCO (13-channel pretrained) but the first
         conv layer is replaced with a 10-channel version.
         The 10 kept conv1 filter slices (one per land band) are copied from the
         pretrained weights; the 3 atmospheric-band slices (B01, B09, B10) are
         discarded.  This reuses pretrained spectral filters for every band we
         actually feed the model, rather than reinitialising all weights from
         scratch when changing the channel count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config


def load_backbone(modality: str) -> nn.Module:
    """
    Load a pretrained ResNet-18 and strip its classification head.

    Returns nn.Sequential outputting (B, 512).  All parameters are unfrozen.
    """
    if modality == "rgb":
        from torchgeo.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights[config.MODALITIES["rgb"]["weights"]])
        return nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    if modality == "ms":
        from torchgeo.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights[config.MODALITIES["ms"]["weights"]])

        # The pretrained conv1 has shape (64, 13, 7, 7) — one filter slice per
        # SSL4EO-S12 canonical band.  We keep only the 10 land-band slices
        # (KEPT_MS_IDX) and drop the 3 atmospheric ones (B01, B09, B10).
        # Atmospheric bands carry no land-use signal; their filters would only
        # add noise to the classification.
        pretrained_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            in_channels  = len(config.KEPT_MS_IDX),  # 10
            out_channels = pretrained_conv1.out_channels,
            kernel_size  = pretrained_conv1.kernel_size,
            stride       = pretrained_conv1.stride,
            padding      = pretrained_conv1.padding,
            bias         = pretrained_conv1.bias is not None,
        )
        # Copy pretrained weights for the 10 kept channels; discard the rest
        new_conv1.weight.data = pretrained_conv1.weight.data[:, config.KEPT_MS_IDX, :, :]

        model.conv1 = new_conv1
        return nn.Sequential(*list(model.children())[:-1], nn.Flatten())

    raise ValueError(f"Unknown modality '{modality}'. Use 'rgb' or 'ms'.")
