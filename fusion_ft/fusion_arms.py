"""
Arm definitions and feature loading for the multi-stream late-fusion pipeline.

Arm matrix
----------
  rgb_ft              RGB fine-tuned CNN (512-d)
  ms_ft               MS fine-tuned CNN (512-d)
  ms_frozen           MS frozen MoCo CNN (512-d)
  ms_handcrafted      MS spectral (32) + texture (22) = 54-d
  ms_fusion           MS ft (512) ⊕ spectral (32) ⊕ texture (22) = 566-d  ← HEADLINE
  ms_fusion_spectral  MS ft (512) ⊕ spectral (32)
  ms_fusion_texture   MS ft (512) ⊕ texture (22)
  ms_fusion_hybrid    RGB ft (512) ⊕ MS spectral (32) ⊕ texture (22)
  ms_fusion_frozen    MS frozen (512) ⊕ spectral (32) ⊕ texture (22)
  rgb_handcrafted     RGB visible indices (12) + grayscale texture (22)
  rgb_fusion          RGB ft (512) ⊕ visible indices (12) ⊕ texture (22)

Each sub-stream is StandardScaler-normalised independently (fitted on train).
Val set is carved from the stored train indices (same as CNN and LGBM training).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from fusion_ft.embeddings import load_ft, load_frozen, verify_label_alignment
from fusion_ft.handcrafted import (
    SPECTRAL_DIM, TEXTURE_DIM, extract as extract_hc,
    RGB_SPECTRAL_DIM, RGB_TEXTURE_DIM, RGB_FULL_DIM, extract_rgb,
)

FUSION_ROOT = config.PROJECT_ROOT / "results_fusion_ft"
SCALER_DIR  = FUSION_ROOT / "scalers"
MODEL_DIR   = FUSION_ROOT / "models"
for _d in (SCALER_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    "rgb_ft":             {"embed": ("rgb", "ft"),     "hc": None},
    "ms_ft":              {"embed": ("ms",  "ft"),     "hc": None},
    "ms_frozen":          {"embed": ("ms",  "frozen"), "hc": None},
    "ms_handcrafted":     {"embed": None,              "hc": "full"},
    "ms_fusion":          {"embed": ("ms",  "ft"),     "hc": "full"},
    "ms_fusion_spectral": {"embed": ("ms",  "ft"),     "hc": "spectral"},
    "ms_fusion_texture":  {"embed": ("ms",  "ft"),     "hc": "texture"},
    "ms_fusion_hybrid":   {"embed": ("rgb", "ft"),     "hc": "full"},
    "ms_fusion_frozen":   {"embed": ("ms",  "frozen"), "hc": "full"},
    "rgb_handcrafted":    {"embed": None,              "hc": "rgb_full"},
    "rgb_fusion":         {"embed": ("rgb", "ft"),     "hc": "rgb_full"},
}

_EMBED_DIM = config.BACKBONE_OUT_DIM  # 512

ARM_DIMS: dict[str, int] = {
    "rgb_ft":             _EMBED_DIM,
    "ms_ft":              _EMBED_DIM,
    "ms_frozen":          _EMBED_DIM,
    "ms_handcrafted":     SPECTRAL_DIM + TEXTURE_DIM,
    "ms_fusion":          _EMBED_DIM + SPECTRAL_DIM + TEXTURE_DIM,
    "ms_fusion_spectral": _EMBED_DIM + SPECTRAL_DIM,
    "ms_fusion_texture":  _EMBED_DIM + TEXTURE_DIM,
    "ms_fusion_hybrid":   _EMBED_DIM + SPECTRAL_DIM + TEXTURE_DIM,
    "ms_fusion_frozen":   _EMBED_DIM + SPECTRAL_DIM + TEXTURE_DIM,
    "rgb_handcrafted":    RGB_FULL_DIM,
    "rgb_fusion":         _EMBED_DIM + RGB_FULL_DIM,
}

# Stream slices within the concatenated vector (for SHAP grouping)
ARM_STREAM_SLICES: dict[str, dict[str, slice]] = {
    "ms_fusion":          {"deep": slice(0, 512), "spectral": slice(512, 544), "texture": slice(544, 566)},
    "ms_fusion_spectral": {"deep": slice(0, 512), "spectral": slice(512, 544)},
    "ms_fusion_texture":  {"deep": slice(0, 512), "texture":  slice(512, 534)},
    "ms_fusion_hybrid":   {"deep": slice(0, 512), "spectral": slice(512, 544), "texture": slice(544, 566)},
    "ms_fusion_frozen":   {"deep": slice(0, 512), "spectral": slice(512, 544), "texture": slice(544, 566)},
    "rgb_fusion":         {"deep": slice(0, 512), "rgb_spectral": slice(512, 524), "rgb_texture": slice(524, 546)},
}


# ---------------------------------------------------------------------------
# Scaler persistence
# ---------------------------------------------------------------------------

def _scaler_path(arm: str) -> Path:
    return SCALER_DIR / f"{arm}.pkl"


def _save_scalers(arm: str, scalers: dict) -> None:
    with open(_scaler_path(arm), "wb") as f:
        pickle.dump(scalers, f)


def _load_scalers(arm: str) -> dict:
    with open(_scaler_path(arm), "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _get_embed(modality: str, source: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    return load_ft(modality, split) if source == "ft" else load_frozen(modality, split)


def load_features(
    arm: str,
    split: str,
    scalers: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build the concatenated feature matrix for *arm* and *split*.

    scalers=None  → fit new scalers on this split (use for train)
    scalers=dict  → apply existing scalers (use for val / test)

    Returns (X, y, scalers).
    """
    cfg   = ARMS[arm]
    parts: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    label_names:  list[str]        = []
    new_sc: dict = {}
    fit_mode = scalers is None

    # Deep stream
    if cfg["embed"] is not None:
        mod, src = cfg["embed"]
        X_e, y_e = _get_embed(mod, src, split)
        label_arrays.append(y_e)
        label_names.append(f"embed_{mod}_{src}")
        if fit_mode:
            sc  = StandardScaler()
            X_e = sc.fit_transform(X_e)
            new_sc["embed"] = sc
        else:
            X_e = scalers["embed"].transform(X_e)
        parts.append(X_e)

    # MS handcrafted stream
    if cfg["hc"] in ("full", "spectral", "texture"):
        X_s, X_t, y_hc = extract_hc(split)
        label_arrays.append(y_hc)
        label_names.append("ms_handcrafted")

        if cfg["hc"] in ("full", "spectral"):
            X_s = X_s.copy()
            if fit_mode:
                sc  = StandardScaler()
                X_s = sc.fit_transform(X_s)
                new_sc["spectral"] = sc
            else:
                X_s = scalers["spectral"].transform(X_s)
            parts.append(X_s)

        if cfg["hc"] in ("full", "texture"):
            X_t = X_t.copy()
            if fit_mode:
                sc  = StandardScaler()
                X_t = sc.fit_transform(X_t)
                new_sc["texture"] = sc
            else:
                X_t = scalers["texture"].transform(X_t)
            parts.append(X_t)

    # RGB handcrafted stream
    elif cfg["hc"] == "rgb_full":
        X_rs, X_rt, y_rgb = extract_rgb(split)
        label_arrays.append(y_rgb)
        label_names.append("rgb_handcrafted")

        X_rs = X_rs.copy()
        if fit_mode:
            sc   = StandardScaler()
            X_rs = sc.fit_transform(X_rs)
            new_sc["rgb_spectral"] = sc
        else:
            X_rs = scalers["rgb_spectral"].transform(X_rs)
        parts.append(X_rs)

        X_rt = X_rt.copy()
        if fit_mode:
            sc   = StandardScaler()
            X_rt = sc.fit_transform(X_rt)
            new_sc["rgb_texture"] = sc
        else:
            X_rt = scalers["rgb_texture"].transform(X_rt)
        parts.append(X_rt)

    if len(label_arrays) > 1:
        verify_label_alignment(*label_arrays, names=label_names)
    y = label_arrays[0]

    X = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    assert X.shape[1] == ARM_DIMS[arm], (
        f"[{arm}/{split}] expected dim {ARM_DIMS[arm]}, got {X.shape[1]}"
    )
    return X.astype(np.float32), y, new_sc if fit_mode else scalers


def carve_val(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified val split from full train features (for LGBM early stopping)."""
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=config.LGBM_VAL_FRAC, random_state=config.SEED
    )
    tr_rel, va_rel = next(sss.split(np.arange(len(y)), y))
    return X[tr_rel], y[tr_rel], X[va_rel], y[va_rel]
