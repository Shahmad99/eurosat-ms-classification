"""Confusion matrix plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion matrix",
    out_path: Path | None = None,
) -> None:
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    short = [c[:9] for c in config.CLASS_NAMES]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            color = "white" if norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  Saved {out_path.name}")
    plt.close(fig)
