"""
Classification head and combined fine-tuning model.

TwoLayerFC    — Linear(512 → 256) → ReLU → Linear(256 → num_classes)
FinetuneModel — backbone + head, both trained end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from models.backbone import load_backbone


class TwoLayerFC(nn.Module):
    def __init__(
        self,
        in_features: int = config.BACKBONE_OUT_DIM,
        hidden: int = config.HEAD_HIDDEN,
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinetuneModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(modality: str) -> FinetuneModel:
    backbone = load_backbone(modality)
    head     = TwoLayerFC()
    return FinetuneModel(backbone, head)


def get_optimizer(model: FinetuneModel, backbone_lr: float, head_lr: float) -> torch.optim.Optimizer:
    """AdamW with discriminative LRs: lower rate for the pretrained backbone."""
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(),     "lr": head_lr},
        ],
        weight_decay=config.FT_WEIGHT_DECAY,
    )
