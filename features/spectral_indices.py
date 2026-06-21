"""
Spectral index features from EuroSAT 13-band patches.

Computes per-patch statistics (mean, std, p10, p90) for each active spectral
index using only land bands.  Atmospheric bands B01/B09/B10 are never used.

Active indices (all 8 on by default):
  NDVI     (B08−B04)/(B08+B04)              vegetation density
  NDRE     (B08−B05)/(B08+B05)              red-edge / crop type
  NDWI     (B03−B08)/(B03+B08)              open water bodies
  MNDWI    (B03−B11)/(B03+B11)              water vs built-up
  NDWI_Gao (B08−B11)/(B08+B11)             vegetation water content
  NDBI     (B11−B08)/(B11+B08)              built-up / impervious surfaces
  BSI      ((B11+B04)−(B08+B02))/…         bare soil
  SAVI     1.5×(B08−B04)/(B08+B04+0.5)    soil-adjusted vegetation

Feature vector length: 8 indices × 4 stats = 32-d (when all 8 are active).

Reads raw 13-band GeoTIFFs directly (file-order band indices) so it works
independently of the CNN data loader's reorder/drop-atmos logic.

Usage:
    python features/spectral_indices.py            # extract train + test
    python features/spectral_indices.py --force    # re-extract
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from data.splits import get_or_create_splits

_EPS = 1e-6

# Raw file-order band indices (B8A is at index 12, not 8)
_B2  = config.BAND_INDEX["B02"]   # 1
_B3  = config.BAND_INDEX["B03"]   # 2
_B4  = config.BAND_INDEX["B04"]   # 3
_B5  = config.BAND_INDEX["B05"]   # 4
_B8  = config.BAND_INDEX["B08"]   # 7
_B11 = config.BAND_INDEX["B11"]   # 10

_INDEX_REGION: dict[str, str] = {
    "NDVI":     "NIR",
    "NDRE":     "red_edge",
    "NDWI":     "NIR",
    "MNDWI":    "SWIR",
    "NDWI_Gao": "SWIR",
    "NDBI":     "SWIR",
    "BSI":      "SWIR",
    "SAVI":     "NIR",
}


def _active() -> list[str]:
    return [k for k in _INDEX_REGION if k in config.ACTIVE_INDICES]


def schema() -> list[dict]:
    """Ordered list of {name, family, region} for every feature in the vector."""
    out = []
    for idx in _active():
        reg = _INDEX_REGION[idx]
        for stat in ("mean", "std", "p10", "p90"):
            out.append({"name": f"{idx}_{stat}", "family": "indices", "region": reg})
    return out


def feature_dim() -> int:
    return len(schema())


def compute(data: np.ndarray) -> np.ndarray:
    """Extract spectral index statistics from a (13, H, W) uint16 patch."""
    r = data.astype(np.float32) / 10_000.0

    b2, b3, b4 = r[_B2], r[_B3], r[_B4]
    b5, b8, b11 = r[_B5], r[_B8], r[_B11]

    def _nd(a, b):
        return (a - b) / (a + b + _EPS)

    maps: dict[str, np.ndarray] = {
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
    for name in _active():
        m = maps[name]
        feats += [
            float(m.mean()),
            float(m.std()),
            float(np.percentile(m, 10)),
            float(np.percentile(m, 90)),
        ]
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch extraction with caching
# ---------------------------------------------------------------------------

def cache_path(split: str) -> Path:
    return config.CACHE_DIR / f"indices_{split}.npz"


def extract(split: str, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for *split*, loading from cache if available."""
    cp = cache_path(split)
    if cp.exists() and not force:
        print(f"  [cache hit] {cp.name}")
        d = np.load(cp)
        return d["features"], d["labels"]

    splits_data = get_or_create_splits()
    idx    = splits_data[split]
    paths  = [splits_data["paths"][i] for i in idx]
    labels = splits_data["labels"][idx].astype(np.int64)

    print(f"  Extracting spectral indices [{split}]  {len(paths)} patches ...")
    t0 = time.time()
    X  = np.vstack([compute(rasterio.open(p).read()) for p in paths])
    print(f"    → {X.shape}  ({time.time()-t0:.1f}s)")

    np.savez_compressed(cp, features=X, labels=labels)
    return X, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached.")
    args = parser.parse_args()

    print(f"Active indices : {_active()}")
    print(f"Feature dim    : {feature_dim()}")
    for split in ("train", "test"):
        extract(split, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
