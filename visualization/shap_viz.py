"""
SHAP attribution and RandomForest permutation importance.

Both are run on the *indices_only* LightGBM model (32-d spectral index features)
so every feature in the plot maps directly to a physical measurement (NDVI = vegetation
density, NDWI = open water, NDBI = built-up impervious surfaces, etc.).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from features.spectral_indices import extract, schema, _active


def _load_indices_model(ml_dir: Path):
    mp = ml_dir / "indices_only.pkl"
    if not mp.exists():
        raise FileNotFoundError(
            f"Model not found: {mp}\n"
            "Run first:  python train_ml.py --arm indices_only"
        )
    with open(mp, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def run_shap(ml_dir: Path) -> None:
    import shap

    model   = _load_indices_model(ml_dir)
    s       = schema()
    names   = [f["name"]   for f in s]
    regions = [f["region"] for f in s]

    X_test, _ = extract("test")

    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_test)

    # Normalise to (N, F, C)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    elif np.array(sv).ndim == 2:
        sv = np.array(sv)[:, :, np.newaxis]
    else:
        sv = np.array(sv)

    n_features = sv.shape[1]
    n_classes  = sv.shape[2]

    # ── Global feature importance bar chart ──────────────────────────────────
    sv_mean = np.abs(sv).mean(axis=(0, 2))
    order   = np.argsort(sv_mean)[::-1]
    top_n   = min(config.SHAP_TOP_N, n_features)
    top_idx = order[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh([names[i] for i in top_idx], sv_mean[top_idx],
            color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.set(xlabel="Mean |SHAP value|",
           title=f"Global feature importance — indices_only  (top {top_n})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(ml_dir / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shap_summary.png")

    # ── Index × class heatmap (group the 4 stats per index) ─────────────────
    active = _active()
    heat   = np.zeros((len(active), n_classes))
    for ii, idx_name in enumerate(active):
        fidx = [j for j, f in enumerate(s) if f["name"].startswith(idx_name + "_")]
        for ci in range(n_classes):
            heat[ii, ci] = np.abs(sv[:, fidx, ci]).mean()

    fig, ax = plt.subplots(figsize=(12, max(4, len(active) * 0.45)))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set(
        xticks=range(n_classes),
        xticklabels=[c[:7] for c in config.CLASS_NAMES],
        yticks=range(len(active)),
        yticklabels=active,
        title="Spectral index × class SHAP importance",
    )
    ax.xaxis.set_tick_params(rotation=30)
    for ii in range(len(active)):
        for ci in range(n_classes):
            ax.text(ci, ii, f"{heat[ii, ci]:.3f}",
                    ha="center", va="center", fontsize=6)
    plt.tight_layout()
    fig.savefig(ml_dir / "shap_index_by_class.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shap_index_by_class.png")

    # Top-20 JSON
    top20 = [
        {"rank": i + 1,
         "feature": names[order[i]],
         "mean_abs_shap": float(sv_mean[order[i]])}
        for i in range(min(20, n_features))
    ]
    with open(ml_dir / "shap_top20.json", "w") as fh:
        json.dump(top20, fh, indent=2)
    print(f"  Saved shap_top20.json")


# ---------------------------------------------------------------------------
# RF permutation importance (cross-check)
# ---------------------------------------------------------------------------

def run_rf_importance(ml_dir: Path) -> None:
    import random, os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    random.seed(config.SEED)
    np.random.seed(config.SEED)

    s     = schema()
    names = [f["name"] for f in s]

    X_tr, y_tr = extract("train")
    X_te, y_te = extract("test")

    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS, n_jobs=1, random_state=config.SEED
    )
    rf.fit(X_tr, y_tr)
    print(f"    RF test accuracy: {rf.score(X_te, y_te):.4f}")

    result  = permutation_importance(rf, X_te, y_te,
                                     n_repeats=config.RF_N_REPEATS,
                                     random_state=config.SEED, n_jobs=1)
    order   = np.argsort(result.importances_mean)[::-1]
    top_n   = min(len(names), 30)
    top_idx = order[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh(
        [names[i] for i in top_idx],
        result.importances_mean[top_idx],
        xerr=[result.importances_std[i] for i in top_idx],
        color="#FF9800", edgecolor="white", linewidth=0.5,
        ecolor="#555", capsize=3,
    )
    ax.set(xlabel="Permutation importance (mean accuracy drop)",
           title="RF permutation importance — spectral indices (cross-check)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(ml_dir / "rf_importance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rf_importance.png")
