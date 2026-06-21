"""
Stratified 80/20 train / test split builder.

The test set is fixed at 20% of all 27 000 patches and never changes between
experiments.  The validation set is NOT stored here; each training script
carves its own val from the 80% train indices so the test set stays clean.

Usage:
    python data/splits.py           # generate and save splits.npz
    python data/splits.py --show    # print per-class split summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config


def build_file_list() -> tuple[list[str], list[int]]:
    """Return (paths, labels) for every .tif file under MS_ROOT."""
    paths, labels = [], []
    for label, cls in enumerate(config.CLASS_NAMES):
        for p in sorted((config.MS_ROOT / cls).glob("*.tif")):
            paths.append(str(p))
            labels.append(label)
    return paths, labels


def make_splits(paths: list[str], labels: list[int], seed: int = config.SEED) -> dict:
    """Stratified 80/20 split.  Returns dict with 'train' and 'test' index arrays."""
    indices = np.arange(len(paths))
    y       = np.array(labels)
    sss     = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - config.TRAIN_RATIO, random_state=seed
    )
    train_idx, test_idx = next(sss.split(indices, y))
    return {"train": train_idx, "test": test_idx}


def save_splits(paths: list[str], labels: list[int], splits: dict) -> None:
    config.SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.SPLIT_FILE,
        paths  = np.array(paths),
        labels = np.array(labels, dtype=np.int64),
        train  = splits["train"],
        test   = splits["test"],
    )
    print(f"Splits saved → {config.SPLIT_FILE}")


def load_splits() -> dict:
    data = np.load(config.SPLIT_FILE, allow_pickle=True)
    return {
        "paths":  list(data["paths"]),
        "labels": data["labels"],
        "train":  data["train"],
        "test":   data["test"],
    }


def get_or_create_splits() -> dict:
    if config.SPLIT_FILE.exists():
        return load_splits()
    paths, labels = build_file_list()
    splits = make_splits(paths, labels)
    save_splits(paths, labels, splits)
    return load_splits()


def _print_summary(data: dict) -> None:
    y = data["labels"]
    print(f"\n{'Split':<8}  {'N':>6}   per-class counts")
    print("-" * 70)
    for split in ("train", "test"):
        idx    = data[split]
        counts = [(y[idx] == c).sum() for c in range(config.NUM_CLASSES)]
        print(f"  {split:<6}  {len(idx):>6}   {counts}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Print summary of existing splits.")
    args = parser.parse_args()

    if args.show:
        _print_summary(load_splits())
        return

    paths, labels = build_file_list()
    print(f"Total files: {len(paths)}")
    splits = make_splits(paths, labels)
    _print_summary({"labels": np.array(labels), **splits})
    save_splits(paths, labels, splits)


if __name__ == "__main__":
    main()
