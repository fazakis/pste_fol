from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = [
    "pr_auc",
    "roc_auc",
    "f1",
    "recall",
    "precision",
    "balanced_accuracy",
    "mcc",
    "accuracy",
]


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    out = {name: float("nan") for name in METRIC_NAMES}
    if len(np.unique(y_true)) >= 2:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    return out
