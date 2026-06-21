"""Training loss and validation accuracy curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    history: list[dict],
    title: str = "Training curves",
    out_path: Path | None = None,
) -> None:
    if not history:
        return

    epochs   = [h["epoch"]      for h in history]
    losses   = [h["train_loss"] for h in history]
    val_accs = [h["val_acc"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, losses, color="#2196F3")
    ax1.set(xlabel="Epoch", ylabel="Train loss", title="Training loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accs, color="#4CAF50")
    best_ep  = epochs[int(max(range(len(val_accs)), key=lambda i: val_accs[i]))]
    best_acc = max(val_accs)
    ax2.axvline(best_ep, color="red", linestyle="--", alpha=0.5,
                label=f"Best ep {best_ep} ({best_acc:.4f})")
    ax2.set(xlabel="Epoch", ylabel="Val accuracy", title="Validation accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"  Saved {out_path.name}")
    plt.close(fig)
