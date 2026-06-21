"""
Deep-stream embedding extraction for the fusion pipeline.

Wraps features/cnn_extractor; adds a one-time MS band-order assertion and a
label-alignment check between streams.

Sources:
  "ft"     — fine-tuned backbone (train.py output, EuroSAT-aware)
  "frozen" — frozen pretrained MoCo backbone (no fine-tuning)

Both return (N, 512) float32 + (N,) int64 labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config

# ---------------------------------------------------------------------------
# Band-order assertion (run once before any MS extraction)
# ---------------------------------------------------------------------------
_CANONICAL = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
]
_FILE_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
]


def assert_ms_band_order() -> None:
    """Verify BAND_REORDER_MS maps file-order → SSL4EO-S12 canonical order."""
    actual = [_FILE_ORDER[i] for i in config.MS_REORDER]
    assert actual == _CANONICAL, (
        f"MS band order mismatch!\n"
        f"  Got:      {actual}\n"
        f"  Expected: {_CANONICAL}\n"
        f"  Check bands.ms_reorder in config.yaml"
    )
    kept = [actual[i] for i in config.KEPT_MS_IDX]
    print(f"  [band-order OK]  kept bands: {kept}")


# ---------------------------------------------------------------------------
# Embedding loaders
# ---------------------------------------------------------------------------

def load_ft(modality: str, split: str, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    from features.cnn_extractor import extract_finetuned
    return extract_finetuned(modality, split, force=force)


def load_frozen(modality: str, split: str, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    from features.cnn_extractor import extract
    return extract(modality, split, force=force)


def verify_label_alignment(*label_arrays: np.ndarray, names: list[str]) -> None:
    ref = label_arrays[0]
    for y, name in zip(label_arrays[1:], names[1:]):
        assert np.array_equal(ref, y), (
            f"Label mismatch between '{names[0]}' and '{name}'!\n"
            f"  First mismatch at index {int(np.where(ref != y)[0][0])}"
        )
    print(f"  [label alignment OK]  {len(ref)} samples  streams: {names}")
