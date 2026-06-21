"""
Reads config.yaml and exposes flat constants for the rest of the codebase.
Edit config.yaml — this file never needs to change.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.resolve()

with open(PROJECT_ROOT / "config.yaml") as _f:
    _C = yaml.safe_load(_f)


def _p(val: str) -> Path:
    p = Path(val)
    return p if p.is_absolute() else PROJECT_ROOT / p


# Project
SEED       = _C["project"]["seed"]
DEVICE_CFG = _C["project"]["device"]

# Paths
DATA_ROOT   = Path(_C["paths"]["data_root"])
MS_ROOT     = Path(_C["paths"]["ms_root"])
SPLIT_FILE  = _p(_C["paths"]["split_file"])
CACHE_DIR   = _p(_C["paths"]["cache_dir"])
RESULTS_DIR = _p(_C["paths"]["results_dir"])

# Data / split
CLASS_NAMES  = _C["data"]["class_names"]
NUM_CLASSES  = len(CLASS_NAMES)
TRAIN_RATIO  = _C["data"]["train_ratio"]
VAL_FRACTION = _C["data"]["val_fraction"]
NUM_WORKERS  = _C["data"]["num_workers"]

# Bands
MS_REORDER        = _C["bands"]["ms_reorder"]
ATMOS_CANONICAL   = _C["bands"]["atmos_canonical_idx"]
KEPT_MS_IDX       = _C["bands"]["kept_ms_idx"]
KEPT_MS_BANDS     = _C["bands"]["kept_ms_bands"]
RGB_FILE_IDX      = _C["bands"]["rgb_file_idx"]
BAND_INDEX        = _C["bands"]["index_map"]

# Derived convenience shortcuts
IDX_B02 = BAND_INDEX["B02"]
IDX_B03 = BAND_INDEX["B03"]
IDX_B04 = BAND_INDEX["B04"]
IDX_B05 = BAND_INDEX["B05"]
IDX_B08 = BAND_INDEX["B08"]
IDX_B11 = BAND_INDEX["B11"]
IDX_B12 = BAND_INDEX["B12"]

# Modality config used by model loader
MODALITIES = {
    "rgb": {"mode": "rgb", "weights": _C["cnn"]["weights"]["rgb"], "in_channels": 3},
    "ms":  {"mode": "ms",  "weights": _C["cnn"]["weights"]["ms"],  "in_channels": len(KEPT_MS_IDX)},
}

# CNN fine-tuning
_cnn = _C["cnn"]
BACKBONE_OUT_DIM = _cnn["backbone_out_dim"]
HEAD_HIDDEN      = _cnn["head_hidden"]
FT_BACKBONE_LR   = _cnn["backbone_lr"]
FT_HEAD_LR       = _cnn["head_lr"]
FT_WEIGHT_DECAY  = _cnn["weight_decay"]
FT_BATCH_SIZE    = _cnn["batch_size"]
FT_EPOCHS        = _cnn["epochs"]
FT_PATIENCE      = _cnn["patience"]
FT_LABEL_SMOOTH  = _cnn["label_smoothing"]
AUGMENT          = _cnn["augment"]
FT_USE_AMP       = _cnn["use_amp"]
EXTRACT_BATCH    = _cnn["extract_batch"]

# Spectral features
_feat = _C["features"]
ACTIVE_INDICES = {k for k, v in _feat["indices"].items() if v}
INDEX_STATS    = _feat["index_stats"]
GLCM_CFG       = _feat["glcm"]
LBP_CFG        = _feat["lbp"]
SOBEL_CFG      = _feat["sobel"]

# LightGBM
_lgbm = _C["lgbm"]
LGBM_N_EST       = _lgbm["n_estimators"]
LGBM_LR          = _lgbm["learning_rate"]
LGBM_LEAVES      = _lgbm["num_leaves"]
LGBM_EARLY_STOP  = _lgbm["early_stopping_rounds"]
LGBM_MIN_CHILD   = _lgbm["min_child_samples"]
LGBM_COL_SAMPLE  = _lgbm["colsample_bytree"]
LGBM_SUBSAMPLE   = _lgbm["subsample"]
LGBM_SUB_FREQ    = _lgbm["subsample_freq"]
LGBM_REG_ALPHA   = _lgbm["reg_alpha"]
LGBM_REG_LAMBDA  = _lgbm["reg_lambda"]
LGBM_VAL_FRAC    = _lgbm["val_fraction"]

# SHAP
_shap = _C["shap"]
SHAP_ENABLED    = _shap["enabled"]
SHAP_TOP_N      = _shap["top_n_features"]
RF_N_ESTIMATORS = _shap["rf_n_estimators"]
RF_N_REPEATS    = _shap["rf_n_repeats"]

# Visualization
VIZ = _C["visualization"]

# Ensure output directories exist at import time
for _d in (CACHE_DIR, RESULTS_DIR / "rgb", RESULTS_DIR / "ms", RESULTS_DIR / "ml"):
    _d.mkdir(parents=True, exist_ok=True)
