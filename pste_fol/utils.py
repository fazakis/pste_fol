from __future__ import annotations

from typing import Dict

import numpy as np

EPS = 1e-12


def to_numpy_xy(X, y):
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if len(X_arr) != len(y_arr):
        raise ValueError(f"X and y have different lengths: {len(X_arr)} != {len(y_arr)}")
    if not set(np.unique(y_arr)).issubset({0, 1}):
        raise ValueError("This reproducibility code expects binary labels encoded as 0/1, with 1 as the positive/minority class")
    return X_arr, y_arr


def class_counts(y) -> Dict[int, int]:
    vals, counts = np.unique(np.asarray(y, dtype=int), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def target_minority_count(y, sampling_strategy: float) -> int:
    counts = class_counts(y)
    n_majority = counts.get(0, 0)
    n_minority = counts.get(1, 0)
    strategy = float(sampling_strategy)
    target = int(strategy * n_majority) if strategy <= 1.0 else int(strategy)
    return max(n_minority, target)


def synthetic_needed(y, sampling_strategy: float) -> int:
    counts = class_counts(y)
    return max(0, target_minority_count(y, sampling_strategy) - counts.get(1, 0))


def normalize01(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < EPS:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def clip01(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(p):
    p = clip01(p)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def positive_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X), dtype=float)
        classes = getattr(clf, "classes_", np.array([0, 1]))
        pos_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else min(1, proba.shape[1] - 1)
        return np.asarray(proba[:, pos_idx], dtype=float)
    if hasattr(clf, "decision_function"):
        score = np.asarray(clf.decision_function(X), dtype=float)
        lo, hi = np.nanmin(score), np.nanmax(score)
        if hi - lo <= EPS:
            return np.full(len(score), 0.5)
        return (score - lo) / (hi - lo)
    return np.asarray(clf.predict(X), dtype=float)


def precision_at_top_k(y_true, y_score, k: int | None = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if k is None:
        k = int(np.sum(y_true == 1))
    k = int(min(max(1, k), len(y_true))) if len(y_true) else 0
    if k <= 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")[:k]
    return float(np.mean(y_true[order] == 1))
