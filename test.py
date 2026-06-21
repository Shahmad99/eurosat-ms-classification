"""
Evaluate a trained checkpoint on the held-out 20% test set.

Outputs saved to results/<modality>/
    metrics.json       — overall accuracy, macro-F1, per-class breakdown
    metrics.csv        — same as a table
    confusion.png      — confusion matrix (--visualize)
    training_curves.png — loss/acc curves from the checkpoint history (--visualize)
    sample_gallery.png  — test patches with predicted vs true labels (--visualize)

Usage
-----
  python test.py --modality rgb --checkpoint results/rgb/best.pt
  python test.py --modality ms  --checkpoint results/ms/best.pt --visualize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import settings as config
from data.dataset import make_loader, EuroSATDataset
from data.splits import get_or_create_splits
from data.transforms import get_transform
from evaluation.metrics import compute_metrics
from models.classifier import build_model


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


def load_checkpoint(checkpoint: Path, modality: str) -> tuple[torch.nn.Module, dict]:
    ckpt  = torch.load(checkpoint, map_location="cpu")
    model = build_model(modality)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_pred_list = [], []
    for imgs, labels in loader:
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        y_pred_list.append(preds)
        y_true_list.append(labels.numpy())
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def get_test_loader(modality: str, batch_size: int) -> tuple[object, list, list]:
    """Return (loader, test_paths, test_labels) for the 20% test split."""
    splits     = get_or_create_splits()
    all_paths  = splits["paths"]
    all_labels = splits["labels"]
    test_idx   = splits["test"]

    paths  = [all_paths[i]  for i in test_idx]
    labels = all_labels[test_idx].tolist()
    loader = make_loader(paths, labels, modality, get_transform(augment=False),
                         batch_size, shuffle=False)
    return loader, paths, labels


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_metrics(m: dict, out_dir: Path) -> None:
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    rows = [{"class": "OVERALL", "accuracy": m["accuracy"],
             "macro_f1": m["macro_f1"], "precision": None,
             "recall": None, "f1": None, "support": None}]
    for cls, v in m["per_class"].items():
        rows.append({"class": cls, "accuracy": None, "macro_f1": None, **v})
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def run_visualizations(
    modality: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: list[dict],
    test_paths: list[str],
    test_labels: list[int],
    out_dir: Path,
) -> None:
    from visualization.confusion import plot_confusion_matrix
    from visualization.curves import plot_training_curves

    plot_confusion_matrix(
        y_true, y_pred,
        title    = f"Confusion matrix — {modality.upper()} fine-tuned",
        out_path = out_dir / "confusion.png",
    )

    if history:
        plot_training_curves(
            history,
            title    = f"Training curves — {modality.upper()}",
            out_path = out_dir / "training_curves.png",
        )

    if config.VIZ.get("sample_gallery", True):
        from visualization.gallery import plot_sample_gallery
        plot_sample_gallery(
            paths      = test_paths,
            true_labels  = test_labels,
            pred_labels  = y_pred.tolist(),
            modality   = modality,
            n          = config.VIZ.get("gallery_n", 12),
            out_path   = out_dir / "sample_gallery.png",
        )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(modality: str, checkpoint: Path, batch_size: int, visualize: bool) -> dict:
    device = get_device()
    model, ckpt = load_checkpoint(checkpoint, modality)
    model = model.to(device)

    print(f"  Checkpoint : {checkpoint}")
    best_val = ckpt.get("best_val_acc", float("nan"))
    print(f"  Best val acc (from training) : {best_val:.4f}")

    loader, test_paths, test_labels = get_test_loader(modality, batch_size)

    print(f"  Running inference on {len(test_paths)} test patches ...")
    y_true, y_pred = predict(model, loader, device)

    m = compute_metrics(y_true, y_pred)
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Macro-F1  : {m['macro_f1']:.4f}")
    print()
    for cls, v in m["per_class"].items():
        print(f"    {cls:<25}  P={v['precision']:.3f}  R={v['recall']:.3f}"
              f"  F1={v['f1']:.3f}  n={v['support']}")

    out_dir = checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(m, out_dir)
    print(f"\n  Metrics saved → {out_dir}/")

    if visualize:
        history = ckpt.get("history", [])
        run_visualizations(modality, y_true, y_pred, history,
                           test_paths, test_labels, out_dir)
        print(f"  Visualizations saved → {out_dir}/")

    return m


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the EuroSAT test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--modality",   choices=["rgb", "ms"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to best.pt saved by train.py.")
    parser.add_argument("--batch-size", type=int, default=config.FT_BATCH_SIZE)
    parser.add_argument("--data-path",  type=str, default=None,
                        help="Override MS_ROOT in config.yaml.")
    parser.add_argument("--visualize",  action="store_true",
                        help="Generate confusion matrix, training curves, and sample gallery.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_path:
        config.MS_ROOT = Path(args.data_path)

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\n"
                 f"Run: python train.py --modality {args.modality}")

    print(f"\n{'='*55}")
    print(f"  Test   modality={args.modality.upper()}  checkpoint={args.checkpoint}")
    print(f"{'='*55}\n")

    evaluate(
        modality   = args.modality,
        checkpoint = args.checkpoint,
        batch_size = args.batch_size,
        visualize  = args.visualize,
    )


if __name__ == "__main__":
    main()
