"""
Evaluation metrics: overall accuracy, macro-F1, per-class breakdown, McNemar's test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent.parent))
import settings as config


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc      = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   int(sup[i]),
        }
        for i, cls in enumerate(config.CLASS_NAMES)
    }
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class}


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    label_a: str = "model_a",
    label_b: str = "model_b",
    correction: bool = True,
) -> dict:
    """McNemar's test with Yates continuity correction (default on)."""
    ca = y_pred_a == y_true
    cb = y_pred_b == y_true
    b  = int(np.sum(ca & ~cb))
    c  = int(np.sum(~ca & cb))
    if b + c == 0:
        stat, p = 0.0, 1.0
    else:
        diff = max(abs(b - c) - (1.0 if correction else 0.0), 0.0)
        stat = float(diff ** 2 / (b + c))
        p    = float(1.0 - chi2.cdf(stat, df=1))
    return {
        "model_a": label_a, "model_b": label_b,
        "b_correct_a_wrong_b": b,
        "c_wrong_a_correct_b": c,
        "chi2": stat,
        "p_value": p,
        "significant": p < 0.05,
        "correction": "Yates" if correction else "none",
    }
