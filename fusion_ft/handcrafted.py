"""
Handcrafted feature streams for the late-fusion pipeline.

MS streams (54-d total)
-----------------------
spectral  32-d   8 indices × {mean, std, p10, p90}
                 NDVI, NDRE, NDWI, MNDWI, NDWI_Gao, NDBI, BSI, SAVI
                 All require NIR (B08) or SWIR (B11/B12) — impossible from RGB.
texture   22-d   GLCM on B08 (5 props × 2 dist=10) + LBP (10) + Sobel (2)

RGB streams (34-d total)
-------------------------
spectral  12-d   3 visible-only indices × {mean, std, p10, p90}
                 VARI, ExG, GRVI  (no NIR/SWIR available)
texture   22-d   Same GLCM/LBP/Sobel on grayscale(B02+B03+B04)/3

All computed on native 64×64 reflectance (DN÷10000) in raw file order.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from data.splits import get_or_create_splits

HC_DIR = config.PROJECT_ROOT / "results_fusion_ft" / "features"
HC_DIR.mkdir(parents=True, exist_ok=True)

# Raw file-order band indices
_B2, _B3, _B4, _B5, _B8, _B11 = 1, 2, 3, 4, 7, 10
_EPS = 1e-6

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
_SPECTRAL_INDICES = ("NDVI", "NDRE", "NDWI", "MNDWI", "NDWI_Gao", "NDBI", "BSI", "SAVI")
_SPECTRAL_STATS   = ("mean", "std", "p10", "p90")
_GLCM_PROPS       = ("contrast", "homogeneity", "energy", "correlation", "dissimilarity")
_GLCM_DISTS       = (1, 2)

SPECTRAL_NAMES: list[str] = [
    f"{idx}_{stat}" for idx in _SPECTRAL_INDICES for stat in _SPECTRAL_STATS
]  # 32

TEXTURE_NAMES: list[str] = (
    [f"glcm_{p}_d{d}" for p in _GLCM_PROPS for d in _GLCM_DISTS] +  # 10
    [f"lbp_bin{i}" for i in range(10)] +                              # 10
    ["sobel_mean", "sobel_std"]                                        #  2
)  # 22

SPECTRAL_DIM = len(SPECTRAL_NAMES)   # 32
TEXTURE_DIM  = len(TEXTURE_NAMES)    # 22
FULL_DIM     = SPECTRAL_DIM + TEXTURE_DIM  # 54

_RGB_SPECTRAL_INDICES = ("VARI", "ExG", "GRVI")
RGB_SPECTRAL_NAMES: list[str] = [
    f"{idx}_{stat}" for idx in _RGB_SPECTRAL_INDICES for stat in _SPECTRAL_STATS
]  # 12
RGB_TEXTURE_NAMES: list[str] = (
    [f"rgb_glcm_{p}_d{d}" for p in _GLCM_PROPS for d in _GLCM_DISTS] +
    [f"rgb_lbp_bin{i}" for i in range(10)] +
    ["rgb_sobel_mean", "rgb_sobel_std"]
)  # 22

RGB_SPECTRAL_DIM = len(RGB_SPECTRAL_NAMES)   # 12
RGB_TEXTURE_DIM  = len(RGB_TEXTURE_NAMES)    # 22
RGB_FULL_DIM     = RGB_SPECTRAL_DIM + RGB_TEXTURE_DIM  # 34


def spectral_schema() -> list[dict]:
    out = []
    for idx in _SPECTRAL_INDICES:
        for stat in _SPECTRAL_STATS:
            out.append({"name": f"{idx}_{stat}", "stream": "spectral"})
    return out


def texture_schema() -> list[dict]:
    return [{"name": n, "stream": "texture"} for n in TEXTURE_NAMES]


def full_schema() -> list[dict]:
    return spectral_schema() + texture_schema()


def rgb_spectral_schema() -> list[dict]:
    return [{"name": n, "stream": "spectral_rgb"} for n in RGB_SPECTRAL_NAMES]


def rgb_texture_schema() -> list[dict]:
    return [{"name": n, "stream": "texture_rgb"} for n in RGB_TEXTURE_NAMES]


def rgb_full_schema() -> list[dict]:
    return rgb_spectral_schema() + rgb_texture_schema()


# ---------------------------------------------------------------------------
# Per-patch helpers
# ---------------------------------------------------------------------------

def _quantize(band: np.ndarray, n: int = 64) -> np.ndarray:
    lo, hi = float(band.min()), float(band.max())
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    return ((band - lo) / (hi - lo) * (n - 1)).clip(0, n - 1).astype(np.uint8)


def _glcm_features(band: np.ndarray) -> list[float]:
    import math
    from skimage.feature import graycomatrix, graycoprops
    q   = _quantize(band, 64)
    mat = graycomatrix(
        q, distances=list(_GLCM_DISTS),
        angles=[0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4],
        levels=64, symmetric=True, normed=True,
    )
    return [v for p in _GLCM_PROPS for v in graycoprops(mat, p).mean(axis=1).tolist()]


def _lbp_features(band: np.ndarray) -> list[float]:
    from skimage.feature import local_binary_pattern
    q    = _quantize(band, 256)
    lbp  = local_binary_pattern(q, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist.tolist()


def _sobel_features(band: np.ndarray) -> list[float]:
    from skimage.filters import sobel
    e = sobel(band.astype(np.float64))
    return [float(e.mean()), float(e.std())]


# ---------------------------------------------------------------------------
# Per-patch computation — MS
# ---------------------------------------------------------------------------

def compute_spectral(data: np.ndarray) -> np.ndarray:
    r = data.astype(np.float32) / 10_000.0
    b2, b3, b4, b5, b8, b11 = r[_B2], r[_B3], r[_B4], r[_B5], r[_B8], r[_B11]

    def _nd(a, b): return (a - b) / (a + b + _EPS)

    maps = {
        "NDVI":     _nd(b8,  b4),
        "NDRE":     _nd(b8,  b5),
        "NDWI":     _nd(b3,  b8),
        "MNDWI":    _nd(b3,  b11),
        "NDWI_Gao": _nd(b8,  b11),
        "NDBI":     _nd(b11, b8),
        "BSI":      ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + _EPS),
        "SAVI":     1.5 * (b8 - b4) / (b8 + b4 + 0.5),
    }
    feats: list[float] = []
    for name in _SPECTRAL_INDICES:
        m = maps[name]
        feats += [float(m.mean()), float(m.std()),
                  float(np.percentile(m, 10)), float(np.percentile(m, 90))]
    return np.array(feats, dtype=np.float32)


def compute_texture(data: np.ndarray) -> np.ndarray:
    r  = data.astype(np.float32) / 10_000.0
    b8 = r[_B8]
    return np.array(
        _glcm_features(b8) + _lbp_features(b8) + _sobel_features(b8),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Per-patch computation — RGB
# ---------------------------------------------------------------------------

def compute_spectral_rgb(data: np.ndarray) -> np.ndarray:
    r = data.astype(np.float32) / 10_000.0
    b2, b3, b4 = r[_B2], r[_B3], r[_B4]
    maps = {
        "VARI": (b3 - b4) / (b3 + b4 - b2 + _EPS),
        "ExG":  2.0 * b3 - b4 - b2,
        "GRVI": (b3 - b4) / (b3 + b4 + _EPS),
    }
    feats: list[float] = []
    for name in _RGB_SPECTRAL_INDICES:
        m = maps[name]
        feats += [float(m.mean()), float(m.std()),
                  float(np.percentile(m, 10)), float(np.percentile(m, 90))]
    return np.array(feats, dtype=np.float32)


def compute_texture_rgb(data: np.ndarray) -> np.ndarray:
    r    = data.astype(np.float32) / 10_000.0
    gray = (r[_B2] + r[_B3] + r[_B4]) / 3.0
    return np.array(
        _glcm_features(gray) + _lbp_features(gray) + _sobel_features(gray),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Batch extraction + caching
# ---------------------------------------------------------------------------

def _extract_batch(split: str, compute_fn_s, compute_fn_t, cache_file: Path
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_file.exists():
        print(f"  [cache hit] {cache_file.name}")
        d = np.load(cache_file)
        return d["spectral"], d["texture"], d["labels"]

    splits_data = get_or_create_splits()
    idx    = splits_data[split]
    paths  = [splits_data["paths"][i] for i in idx]
    labels = splits_data["labels"][idx].astype(np.int64)

    print(f"  Extracting handcrafted [{split}]  {len(paths)} patches ...")
    t0 = time.time()
    spec_feats, tex_feats = [], []
    for p in paths:
        with rasterio.open(p) as src:
            data = src.read()
        spec_feats.append(compute_fn_s(data))
        tex_feats.append(compute_fn_t(data))

    X_s = np.vstack(spec_feats).astype(np.float32)
    X_t = np.vstack(tex_feats).astype(np.float32)
    print(f"    → spectral={X_s.shape}  texture={X_t.shape}  ({time.time()-t0:.1f}s)")
    np.savez_compressed(cache_file, spectral=X_s, texture=X_t, labels=labels)
    return X_s, X_t, labels


def extract(split: str, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MS handcrafted features.  Returns (X_spectral, X_texture, y)."""
    cp = HC_DIR / f"ms_hc_{split}.npz"
    if force and cp.exists():
        cp.unlink()
    return _extract_batch(split, compute_spectral, compute_texture, cp)


def extract_rgb(split: str, force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB handcrafted features.  Returns (X_spectral, X_texture, y)."""
    cp = HC_DIR / f"rgb_hc_{split}.npz"
    if force and cp.exists():
        cp.unlink()
    return _extract_batch(split, compute_spectral_rgb, compute_texture_rgb, cp)
