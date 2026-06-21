"""
Sample gallery: a grid of test patches showing predicted vs true label.

Renders the RGB composite (B04/B03/B02) regardless of modality so the patches
are human-readable.  An optional NDVI/NDWI overlay can be added for the MS arm
to support the subjective assessment section of the assignment.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config

# File-order indices for RGB display (B04=red, B03=green, B02=blue)
_R, _G, _B = 3, 2, 1


def _load_rgb(path: str) -> np.ndarray:
    """Load a patch and return a (H, W, 3) uint8 array scaled for display."""
    with rasterio.open(path) as src:
        data = src.read()
    rgb = data[[_R, _G, _B]].astype(np.float32)
    # Percentile stretch: makes patches visually clear even at low reflectance
    for c in range(3):
        lo, hi = np.percentile(rgb[c], 2), np.percentile(rgb[c], 98)
        if hi > lo:
            rgb[c] = (rgb[c] - lo) / (hi - lo)
    rgb = rgb.clip(0, 1).transpose(1, 2, 0)
    return (rgb * 255).astype(np.uint8)


def plot_sample_gallery(
    paths: list[str],
    true_labels: list[int],
    pred_labels: list[int],
    modality: str,
    n: int = 12,
    out_path: Path | None = None,
    seed: int = config.SEED,
) -> None:
    random.seed(seed)
    indices  = list(range(len(paths)))
    # Balance: show roughly equal correct and incorrect predictions
    correct   = [i for i in indices if true_labels[i] == pred_labels[i]]
    incorrect = [i for i in indices if true_labels[i] != pred_labels[i]]
    n_wrong   = min(n // 3, len(incorrect))   # ~1/3 of gallery shows mistakes
    n_right   = min(n - n_wrong, len(correct))
    chosen    = random.sample(incorrect, n_wrong) + random.sample(correct, n_right)
    random.shuffle(chosen)
    chosen    = chosen[:n]

    ncols = 4
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, chosen):
        img  = _load_rgb(paths[idx])
        true = config.CLASS_NAMES[true_labels[idx]]
        pred = config.CLASS_NAMES[pred_labels[idx]]
        correct_pred = true_labels[idx] == pred_labels[idx]

        ax.imshow(img)
        ax.axis("off")
        border_color = "#4CAF50" if correct_pred else "#F44336"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

        label_text = f"T: {true[:12]}\nP: {pred[:12]}"
        ax.set_title(label_text, fontsize=7, pad=2,
                     color=border_color if not correct_pred else "black")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    correct_patch   = mpatches.Patch(color="#4CAF50", label="Correct prediction")
    incorrect_patch = mpatches.Patch(color="#F44336", label="Incorrect prediction")
    fig.legend(handles=[correct_patch, incorrect_patch],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0))

    fig.suptitle(f"Sample test patches — {modality.upper()} model  (RGB display)",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  Saved {out_path.name}")
    plt.close(fig)
