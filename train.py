"""
Fine-tune a pretrained ResNet-18 on EuroSAT.

All hyperparameter defaults come from config.yaml; every flag overrides one value,
so a reviewer can reproduce any run without touching a file.

Architecture
------------
  Backbone  : ResNet-18 pretrained on Sentinel-2 (MoCo)
              RGB → 3-channel input  |  MS → 10 land-band input
  Head      : Linear(512→256) → ReLU → Linear(256→10)
  Training  : AdamW with discriminative LRs, AMP on CUDA, early stopping on val acc

Usage
-----
  python train.py --modality rgb
  python train.py --modality ms --epochs 30 --batch-size 64
  python train.py --modality rgb --smoke-test         # 2 epochs sanity check
  python train.py --modality ms --force               # retrain even if checkpoint exists
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
import settings as config
from data.dataset import make_loader
from data.splits import get_or_create_splits
from data.transforms import get_transform
from models.classifier import build_model, get_optimizer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if config.DEVICE_CFG == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.DEVICE_CFG)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(modality: str, out_dir: Path) -> Path:
    return out_dir / "best.pt"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def make_loaders(modality: str, batch_size: int, augment: bool, val_fraction: float):
    """
    Build train / val DataLoaders by carving val from the stored train split.

    The test split is intentionally excluded here — test.py loads it separately
    from a checkpoint, keeping training and evaluation independent.
    """
    splits     = get_or_create_splits()
    all_paths  = splits["paths"]
    all_labels = splits["labels"]
    tr_idx     = splits["train"]

    # Carve a stratified val set from the 80% train indices
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=config.SEED
    )
    tr_rel, val_rel = next(sss.split(np.arange(len(tr_idx)), all_labels[tr_idx]))
    train_idx = tr_idx[tr_rel]
    val_idx   = tr_idx[val_rel]

    def _paths(idx):  return [all_paths[i] for i in idx]
    def _labels(idx): return all_labels[idx].tolist()

    train_loader = make_loader(
        _paths(train_idx), _labels(train_idx), modality,
        get_transform(augment=augment), batch_size, shuffle=True, drop_last=True,
    )
    val_loader = make_loader(
        _paths(val_idx), _labels(val_idx), modality,
        get_transform(augment=False), batch_size, shuffle=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp) -> float:
    model.train()
    total_loss = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        total      += len(labels)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total   += len(labels)
    return correct / total


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    modality: str,
    epochs: int,
    batch_size: int,
    backbone_lr: float,
    head_lr: float,
    val_fraction: float,
    augment: bool,
    use_amp: bool,
    patience: int,
    label_smoothing: float,
    out_dir: Path,
    force: bool = False,
    smoke_test: bool = False,
) -> list[dict]:
    cp = checkpoint_path(modality, out_dir)
    if cp.exists() and not force:
        print(f"  Checkpoint exists: {cp}  (pass --force to retrain)")
        return torch.load(cp, map_location="cpu").get("history", [])

    seed_everything(config.SEED)
    device  = get_device()
    use_amp = use_amp and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"  Device        : {device}   AMP : {use_amp}")
    print(f"  Backbone LR   : {backbone_lr}   Head LR : {head_lr}")
    print(f"  Batch size    : {batch_size}   Max epochs : {epochs}")

    model     = build_model(modality).to(device)
    optimizer = get_optimizer(model, backbone_lr=backbone_lr, head_lr=head_lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    train_loader, val_loader = make_loaders(modality, batch_size, augment, val_fraction)
    max_ep = 2 if smoke_test else epochs

    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    best_epoch   = 0
    history: list[dict] = []

    for ep in range(max_ep):
        t0      = time.time()
        loss    = train_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            best_epoch   = ep + 1
        else:
            patience_ctr += 1

        history.append({"epoch": ep + 1, "train_loss": loss, "val_acc": val_acc})
        marker = " ✓" if improved else f"  (patience {patience_ctr}/{patience})"
        print(f"  ep {ep+1:3d}/{max_ep}  loss={loss:.4f}  val_acc={val_acc:.4f}"
              f"  [{elapsed:.0f}s]{marker}")

        if patience_ctr >= patience:
            print(f"  Early stop — best epoch {best_epoch}, val_acc={best_val_acc:.4f}")
            break

    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "modality":     modality,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
        "model_state":  model.state_dict(),
        "history":      history,
    }, cp)
    print(f"  Checkpoint saved → {cp}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-18 on EuroSAT (RGB or MS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modality",      choices=["rgb", "ms"], required=True)
    parser.add_argument("--epochs",        type=int,   default=config.FT_EPOCHS)
    parser.add_argument("--batch-size",    type=int,   default=config.FT_BATCH_SIZE)
    parser.add_argument("--backbone-lr",   type=float, default=config.FT_BACKBONE_LR)
    parser.add_argument("--head-lr",       type=float, default=config.FT_HEAD_LR)
    parser.add_argument("--patience",      type=int,   default=config.FT_PATIENCE)
    parser.add_argument("--data-path",     type=str,   default=None,
                        help="Override MS_ROOT in config.yaml.")
    parser.add_argument("--results-dir",   type=str,   default=None,
                        help="Override results directory.")
    parser.add_argument("--force",         action="store_true",
                        help="Retrain even if a checkpoint already exists.")
    parser.add_argument("--smoke-test",    action="store_true",
                        help="Run 2 epochs only — sanity check.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_path:
        config.MS_ROOT = Path(args.data_path)

    out_dir = Path(args.results_dir) / args.modality if args.results_dir else \
              config.RESULTS_DIR / args.modality

    print(f"\n{'='*55}")
    print(f"  Train  modality={args.modality.upper()}  epochs={args.epochs}"
          f"  batch={args.batch_size}")
    print(f"{'='*55}\n")

    train(
        modality        = args.modality,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        backbone_lr     = args.backbone_lr,
        head_lr         = args.head_lr,
        val_fraction    = config.VAL_FRACTION,
        augment         = config.AUGMENT,
        use_amp         = config.FT_USE_AMP,
        patience        = args.patience,
        label_smoothing = config.FT_LABEL_SMOOTH,
        out_dir         = out_dir,
        force           = args.force,
        smoke_test      = args.smoke_test,
    )


if __name__ == "__main__":
    main()
