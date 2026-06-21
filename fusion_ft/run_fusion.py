"""
Multi-stream late-fusion training, evaluation, SHAP attribution, and reporting.

Usage:
    python fusion_ft/run_fusion.py                  # full pipeline
    python fusion_ft/run_fusion.py --force          # recompute all caches / models
    python fusion_ft/run_fusion.py --embed-only     # extract embeddings and stop
    python fusion_ft/run_fusion.py --hc-only        # extract handcrafted features and stop
    python fusion_ft/run_fusion.py --arm ms_fusion  # single arm
    python fusion_ft/run_fusion.py --no-xgb         # skip XGBoost comparison

Outputs — all in results_fusion_ft/
    embeddings/        fine-tuned and frozen 512-d caches
    features/          handcrafted 54-d caches
    models/            .pkl models per arm × classifier
    scalers/           StandardScaler dicts per arm
    confusion_<arm>.png
    shap_stream_by_class.png  (ms_fusion)
    shap_summary.png
    shap_top20.json
    metrics.json / metrics.csv
    summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config
from evaluation.metrics import compute_metrics, mcnemar_test
from fusion_ft.embeddings import assert_ms_band_order
from fusion_ft.fusion_arms import (
    ARMS, ARM_DIMS, ARM_STREAM_SLICES,
    FUSION_ROOT, MODEL_DIR, SCALER_DIR,
    _load_scalers, _save_scalers,
    load_features, carve_val,
)
from fusion_ft.handcrafted import (
    SPECTRAL_NAMES, TEXTURE_NAMES,
    extract as extract_hc,
    extract_rgb,
)
from visualization.confusion import plot_confusion_matrix

FUSION_ROOT.mkdir(parents=True, exist_ok=True)


def _seed() -> None:
    random.seed(config.SEED)
    os.environ["PYTHONHASHSEED"] = str(config.SEED)
    np.random.seed(config.SEED)


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

def lgbm_path(arm: str) -> Path:
    return MODEL_DIR / f"lgbm_{arm}.pkl"


def xgb_path(arm: str) -> Path:
    return MODEL_DIR / f"xgb_{arm}.pkl"


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def _train_lgbm(X_tr, y_tr, X_va, y_va):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators      = config.LGBM_N_EST,
        learning_rate     = config.LGBM_LR,
        num_leaves        = config.LGBM_LEAVES,
        min_child_samples = config.LGBM_MIN_CHILD,
        colsample_bytree  = config.LGBM_COL_SAMPLE,
        subsample         = config.LGBM_SUBSAMPLE,
        subsample_freq    = config.LGBM_SUB_FREQ,
        reg_alpha         = config.LGBM_REG_ALPHA,
        reg_lambda        = config.LGBM_REG_LAMBDA,
        random_state      = config.SEED,
        n_jobs            = 1,
        verbose           = -1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.LGBM_EARLY_STOP, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def _train_xgb(X_tr, y_tr, X_va, y_va):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators          = config.LGBM_N_EST,
        learning_rate         = config.LGBM_LR,
        max_depth             = 6,
        min_child_weight      = config.LGBM_MIN_CHILD,
        subsample             = config.LGBM_SUBSAMPLE,
        colsample_bytree      = config.LGBM_COL_SAMPLE,
        reg_alpha             = config.LGBM_REG_ALPHA,
        reg_lambda            = config.LGBM_REG_LAMBDA,
        early_stopping_rounds = config.LGBM_EARLY_STOP,
        random_state          = config.SEED,
        n_jobs                = 1,
        verbosity             = 0,
        eval_metric           = "mlogloss",
        tree_method           = "hist",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Train one arm
# ---------------------------------------------------------------------------

def train_arm(arm: str, run_xgb: bool = True, force: bool = False) -> None:
    lp = lgbm_path(arm)
    sp = SCALER_DIR / f"{arm}.pkl"

    if lp.exists() and sp.exists() and not force:
        print(f"  [cache hit] {lp.name}")
        return

    _seed()
    t0 = time.time()

    # Load all train features; scalers fitted on train only
    X_tr_full, y_tr_full, scalers = load_features(arm, "train", scalers=None)
    X_tr, y_tr, X_va, y_va        = carve_val(X_tr_full, y_tr_full)

    print(f"  [{arm}]  train={X_tr.shape}  val={X_va.shape}  dim={ARM_DIMS[arm]}")

    lgbm_model = _train_lgbm(X_tr, y_tr, X_va, y_va)
    best = lgbm_model.best_iteration_ or lgbm_model.n_estimators
    print(f"    LGBM best_iter={best}  "
          f"val_acc={(lgbm_model.predict(X_va)==y_va).mean():.4f}  "
          f"({time.time()-t0:.0f}s)")

    with open(lp, "wb") as f:
        pickle.dump(lgbm_model, f)
    _save_scalers(arm, scalers)

    if run_xgb and not (xgb_path(arm).exists() and not force):
        t1 = time.time()
        xgb_model = _train_xgb(X_tr, y_tr, X_va, y_va)
        print(f"    XGB  val_acc={(xgb_model.predict(X_va)==y_va).mean():.4f}  "
              f"({time.time()-t1:.0f}s)")
        with open(xgb_path(arm), "wb") as f:
            pickle.dump(xgb_model, f)


# ---------------------------------------------------------------------------
# Evaluate one arm
# ---------------------------------------------------------------------------

def evaluate_arm(arm: str, classifier: str = "lgbm") -> tuple[np.ndarray, np.ndarray, dict]:
    mp = lgbm_path(arm) if classifier == "lgbm" else xgb_path(arm)
    if not mp.exists():
        raise FileNotFoundError(f"No {classifier} model for '{arm}'. Train first.")
    with open(mp, "rb") as f:
        model = pickle.load(f)
    scalers = _load_scalers(arm)
    X_te, y_te, _ = load_features(arm, "test", scalers=scalers)
    y_pred = model.predict(X_te)
    m      = compute_metrics(y_te, y_pred)
    print(f"  [{arm:<22}] {classifier.upper()} | "
          f"acc={m['accuracy']:.4f}  F1={m['macro_f1']:.4f}  dim={ARM_DIMS[arm]}")
    return y_te, y_pred, m


# ---------------------------------------------------------------------------
# SHAP on ms_fusion
# ---------------------------------------------------------------------------

def run_shap(out_dir: Path) -> None:
    import shap

    arm = "ms_fusion"
    print(f"\n  Running SHAP on [{arm}] ...")
    with open(lgbm_path(arm), "rb") as f:
        model = pickle.load(f)
    scalers = _load_scalers(arm)
    X_te, _, _ = load_features(arm, "test", scalers=scalers)

    rng    = np.random.default_rng(config.SEED)
    n_samp = min(1500, len(X_te))
    idx    = rng.choice(len(X_te), n_samp, replace=False)
    X_samp = X_te[idx]

    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_samp)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    elif np.array(sv).ndim == 2:
        sv = np.array(sv)[:, :, np.newaxis]
    else:
        sv = np.array(sv)

    n_classes  = sv.shape[2]
    slices     = ARM_STREAM_SLICES[arm]
    hc_names   = SPECTRAL_NAMES + TEXTURE_NAMES  # 54 named

    # Handcrafted-only global importance (skip the 512 deep features)
    sv_mean  = np.abs(sv).mean(axis=(0, 2))
    hc_mean  = sv_mean[512:]
    hc_order = np.argsort(hc_mean)[::-1]
    top_n    = min(40, len(hc_names))
    top_idx  = hc_order[:top_n][::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)))
    ax.barh([hc_names[i] for i in top_idx], hc_mean[top_idx],
            color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.set(xlabel="Mean |SHAP value|",
           title=f"ms_fusion: handcrafted feature importance (top {top_n})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shap_summary.png")

    # Stream × class heatmap
    stream_order = list(slices.keys())
    heat = np.zeros((len(stream_order), n_classes))
    for si, stream in enumerate(stream_order):
        sl = slices[stream]
        for ci in range(n_classes):
            heat[si, ci] = np.abs(sv[:, sl, ci]).mean()

    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set(
        xticks=range(n_classes),
        xticklabels=[c[:7] for c in config.CLASS_NAMES],
        yticks=range(len(stream_order)),
        yticklabels=stream_order,
        title="ms_fusion: stream × class SHAP importance",
    )
    ax.xaxis.set_tick_params(rotation=30)
    for si in range(len(stream_order)):
        for ci in range(n_classes):
            ax.text(ci, si, f"{heat[si, ci]:.3f}",
                    ha="center", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_stream_by_class.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved shap_stream_by_class.png")

    top20 = [
        {"rank": i + 1, "feature": hc_names[hc_order[i]],
         "mean_abs_shap": float(hc_mean[hc_order[i]])}
        for i in range(min(20, len(hc_names)))
    ]
    with open(out_dir / "shap_top20.json", "w") as fh:
        json.dump(top20, fh, indent=2)
    print(f"  Saved shap_top20.json")


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"


def build_summary(lgbm_metrics: dict, xgb_metrics: dict, out_dir: Path) -> None:
    def acc(d, arm): return d.get(arm, {}).get("accuracy",  float("nan"))
    def f1(d, arm):  return d.get(arm, {}).get("macro_f1",  float("nan"))

    frozen_lift = acc(lgbm_metrics, "ms_fusion_frozen") - acc(lgbm_metrics, "ms_frozen")
    ft_lift     = acc(lgbm_metrics, "ms_fusion")        - acc(lgbm_metrics, "ms_ft")
    subsumption = frozen_lift - ft_lift

    lines: list[str] = [
        "# Late-Fusion Results — Fine-tuned CNN ⊕ Handcrafted Streams",
        "",
        "## Fusion Matrix (LightGBM)",
        "",
        "| Arm | Deep stream | Handcrafted | Dim | Accuracy | Macro-F1 |",
        "|-----|-------------|-------------|-----|----------|----------|",
    ]
    descs = {
        "rgb_ft":             ("RGB ft",    "—"),
        "ms_ft":              ("MS ft",     "—"),
        "ms_frozen":          ("MS frozen", "—"),
        "ms_handcrafted":     ("—",         "MS spectral + texture"),
        "ms_fusion":          ("MS ft",     "MS spectral + texture  ← **headline**"),
        "ms_fusion_spectral": ("MS ft",     "MS spectral only"),
        "ms_fusion_texture":  ("MS ft",     "MS texture only"),
        "ms_fusion_hybrid":   ("RGB ft",    "MS spectral + texture"),
        "ms_fusion_frozen":   ("MS frozen", "MS spectral + texture"),
        "rgb_handcrafted":    ("—",         "RGB visible indices + grayscale texture"),
        "rgb_fusion":         ("RGB ft",    "RGB visible indices + grayscale texture"),
    }
    for arm in ARMS:
        deep, hc = descs.get(arm, ("?", "?"))
        lines.append(
            f"| `{arm}` | {deep} | {hc} | {ARM_DIMS[arm]} "
            f"| {_fmt(acc(lgbm_metrics, arm))} | {_fmt(f1(lgbm_metrics, arm))} |"
        )

    lines += [
        "",
        "## Key Findings",
        "",
        f"- MS ft alone:          {_fmt(acc(lgbm_metrics,'ms_ft'))}",
        f"- MS handcrafted alone:  {_fmt(acc(lgbm_metrics,'ms_handcrafted'))}",
        f"- MS fusion (headline):  {_fmt(acc(lgbm_metrics,'ms_fusion'))}",
        "",
        f"- Frozen CNN + handcrafted lift:     {frozen_lift:+.4f}",
        f"- Fine-tuned CNN + handcrafted lift: {ft_lift:+.4f}",
        f"- Subsumption delta (frozen−ft lift): {subsumption:+.4f}",
        "",
        "A positive subsumption delta means fine-tuning already learned some of "
        "the spectral contrast the frozen backbone needed explicit indices to capture.",
        "",
        "---",
        "_Auto-generated by `fusion_ft/run_fusion.py`_",
    ]

    out_path = out_dir / "summary.md"
    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(arms: list[str] | None = None, run_xgb: bool = True, force: bool = False) -> None:
    _seed()
    target_arms = arms or list(ARMS.keys())

    print("\n=== Step 1: Band-order verification ===")
    assert_ms_band_order()

    print("\n=== Step 2: Handcrafted feature extraction ===")
    for split in ("train", "test"):
        extract_hc(split, force=force)
    rgb_arms = {"rgb_handcrafted", "rgb_fusion"}
    if not arms or any(a in rgb_arms for a in target_arms):
        for split in ("train", "test"):
            extract_rgb(split, force=force)

    print("\n=== Step 3: Training ===")
    for arm in target_arms:
        train_arm(arm, run_xgb=run_xgb, force=force)

    print("\n=== Step 4: Evaluation (LightGBM, test set) ===")
    lgbm_metrics: dict[str, dict] = {}
    lgbm_preds:   dict[str, tuple] = {}
    for arm in target_arms:
        try:
            y_true, y_pred, m = evaluate_arm(arm, "lgbm")
            lgbm_metrics[arm] = m
            lgbm_preds[arm]   = (y_true, y_pred)
            plot_confusion_matrix(
                y_true, y_pred,
                title    = f"ms_fusion (LGBM) — {arm}",
                out_path = FUSION_ROOT / f"confusion_{arm}.png",
            )
        except FileNotFoundError as e:
            print(f"  [{arm}] skipped: {e}")

    xgb_metrics: dict[str, dict] = {}
    if run_xgb:
        print("\n=== Step 4b: Evaluation (XGBoost, test set) ===")
        for arm in target_arms:
            if xgb_path(arm).exists():
                try:
                    _, _, m = evaluate_arm(arm, "xgb")
                    xgb_metrics[arm] = m
                except FileNotFoundError as e:
                    print(f"  [{arm}] XGB skipped: {e}")

    # McNemar tests
    mcnemar_results: dict = {}
    for a1, a2 in [
        ("ms_ft", "ms_fusion"),
        ("rgb_ft", "ms_ft"),
        ("ms_fusion_frozen", "ms_fusion"),
        ("ms_fusion_spectral", "ms_fusion_texture"),
    ]:
        if a1 in lgbm_preds and a2 in lgbm_preds:
            y_t1, y_p1 = lgbm_preds[a1]
            y_t2, y_p2 = lgbm_preds[a2]
            if np.array_equal(y_t1, y_t2):
                mcnemar_results[f"{a1}_vs_{a2}"] = mcnemar_test(y_t1, y_p1, y_p2, a1, a2)

    # Persist
    with open(FUSION_ROOT / "metrics.json", "w") as f:
        json.dump({"lgbm": lgbm_metrics, "xgb": xgb_metrics,
                   "mcnemar": mcnemar_results}, f, indent=2)

    rows = []
    for clf, mdict in [("lgbm", lgbm_metrics), ("xgb", xgb_metrics)]:
        for arm, m in mdict.items():
            rows.append({"classifier": clf, "arm": arm, "class": "OVERALL",
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "dim": ARM_DIMS.get(arm)})
    pd.DataFrame(rows).to_csv(FUSION_ROOT / "metrics.csv", index=False)
    print(f"\n  Metrics saved → {FUSION_ROOT}/metrics.json")

    if "ms_fusion" in lgbm_preds:
        print("\n=== Step 5: SHAP attribution (ms_fusion) ===")
        run_shap(FUSION_ROOT)

    print("\n=== Step 6: Summary report ===")
    build_summary(lgbm_metrics, xgb_metrics, FUSION_ROOT)

    print(f"\n{'='*55}")
    print(f"  Done.  Results in: {FUSION_ROOT}/")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Late-fusion training + evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arm",        nargs="+", choices=list(ARMS.keys()))
    parser.add_argument("--force",      action="store_true")
    parser.add_argument("--no-xgb",    action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--hc-only",    action="store_true")
    args = parser.parse_args()

    if args.embed_only:
        print("\n=== Embedding extraction only ===")
        assert_ms_band_order()
        from features.cnn_extractor import extract, extract_finetuned
        for mod in ("rgb", "ms"):
            for split in ("train", "test"):
                extract_finetuned(mod, split, force=args.force)
        for split in ("train", "test"):
            extract("ms", split, force=args.force)
        return

    if args.hc_only:
        print("\n=== Handcrafted feature extraction only ===")
        for split in ("train", "test"):
            extract_hc(split, force=args.force)
            extract_rgb(split, force=args.force)
        return

    run(arms=args.arm, run_xgb=not args.no_xgb, force=args.force)


if __name__ == "__main__":
    main()
