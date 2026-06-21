"""
CNN backbone feature extraction for the ML / fusion arms.

extract_finetuned()  Loads best.pt (saved by train.py), strips the
                     classification head, and runs the backbone in eval mode.
                     For MS: the backbone expects 10 land-band input matching
                     what train.py produced (reorder → drop B01/B09/B10).
                     Returns 512-d embeddings that are EuroSAT-aware.

extract()            Frozen pretrained backbone (original MoCo weights, no
                     fine-tuning).  Kept for the frozen-baseline arms in the
                     late-fusion pipeline.

Cache paths:
    cache/cnn_ft_{modality}_{split}.npz      fine-tuned
    cache/cnn_{modality}_{split}.npz         frozen pretrained

Usage:
    python features/cnn_extractor.py --modality rgb --finetuned
    python features/cnn_extractor.py --modality ms  --finetuned
    python features/cnn_extractor.py --modality ms              # frozen
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from data.dataset import EuroSATDataset
from data.splits import get_or_create_splits
from data.transforms import get_transform
from models.backbone import load_backbone


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(modality: str, split: str) -> tuple[DataLoader, np.ndarray]:
    splits_data = get_or_create_splits()
    idx    = splits_data[split]
    paths  = [splits_data["paths"][i] for i in idx]
    labels = splits_data["labels"][idx].astype(np.int64)
    ds     = EuroSATDataset(paths, labels.tolist(), modality=modality,
                            transform=get_transform(augment=False))
    loader = DataLoader(ds, batch_size=config.EXTRACT_BATCH, shuffle=False,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())
    return loader, labels


@torch.no_grad()
def _run_backbone(backbone: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> np.ndarray:
    backbone.eval()
    feats = []
    for imgs, _ in loader:
        feats.append(backbone(imgs.to(device)).cpu().numpy())
    return np.vstack(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# Fine-tuned backbone
# ---------------------------------------------------------------------------

def finetuned_cache_path(modality: str, split: str) -> Path:
    return config.CACHE_DIR / f"cnn_ft_{modality}_{split}.npz"


@torch.no_grad()
def extract_finetuned(
    modality: str,
    split: str,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 512-d features from the fine-tuned backbone (head stripped).

    Requires train.py to have been run for this modality first.
    """
    cp = finetuned_cache_path(modality, split)
    if cp.exists() and not force:
        print(f"  [cache hit] {cp.name}")
        d = np.load(cp)
        return d["features"], d["labels"]

    ckpt_file = config.RESULTS_DIR / modality / "best.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_file}.\n"
            f"Run first:  python train.py --modality {modality}"
        )

    from models.classifier import build_model
    device = get_device()
    model  = build_model(modality)
    ckpt   = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # The backbone already outputs (B, 512); strip the head entirely
    backbone = model.backbone.to(device)
    best_val = ckpt.get("best_val_acc", float("nan"))
    print(f"  Loaded fine-tuned [{modality}]  (best val acc = {best_val:.4f})")

    loader, labels = _make_loader(modality, split)
    print(f"  Extracting fine-tuned CNN features [{modality}/{split}]  "
          f"{len(labels)} patches ...")
    t0 = time.time()
    X  = _run_backbone(backbone, loader, device)
    print(f"    → {X.shape}  ({time.time()-t0:.1f}s)")

    np.savez_compressed(cp, features=X, labels=labels)
    return X, labels


# ---------------------------------------------------------------------------
# Frozen pretrained backbone
# ---------------------------------------------------------------------------

def frozen_cache_path(modality: str, split: str) -> Path:
    return config.CACHE_DIR / f"cnn_{modality}_{split}.npz"


@torch.no_grad()
def extract(
    modality: str,
    split: str,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 512-d features from the frozen pretrained backbone."""
    cp = frozen_cache_path(modality, split)
    if cp.exists() and not force:
        print(f"  [cache hit] {cp.name}")
        d = np.load(cp)
        return d["features"], d["labels"]

    device   = get_device()
    backbone = load_backbone(modality).to(device)
    backbone.eval()

    loader, labels = _make_loader(modality, split)
    print(f"  Extracting frozen CNN features [{modality}/{split}]  "
          f"{len(labels)} patches ...")
    t0 = time.time()
    X  = _run_backbone(backbone, loader, device)
    print(f"    → {X.shape}  ({time.time()-t0:.1f}s)")

    np.savez_compressed(cp, features=X, labels=labels)
    return X, labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--modality", choices=["rgb", "ms", "all"], default="all")
    parser.add_argument("--finetuned", action="store_true",
                        help="Use fine-tuned backbone (requires train.py output).")
    parser.add_argument("--force",    action="store_true")
    args = parser.parse_args()

    _seed()
    mods = ["rgb", "ms"] if args.modality == "all" else [args.modality]
    for mod in mods:
        for split in ("train", "test"):
            if args.finetuned:
                extract_finetuned(mod, split, force=args.force)
            else:
                extract(mod, split, force=args.force)


if __name__ == "__main__":
    main()
