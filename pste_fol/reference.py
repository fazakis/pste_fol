from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference" / "primary"
REFERENCE_METRICS = REFERENCE_DIR / "validated_fold_metrics_17820.csv"

BACKBONES = ("bagged_cart", "randomforest", "extratrees")
SEEDS = (42, 44, 49)
FOLDS = (1, 2, 3, 4, 5)
METRICS = (
    "pr_auc",
    "roc_auc",
    "f1",
    "recall",
    "precision",
    "balanced_accuracy",
    "mcc",
    "accuracy",
)
METHODS = (
    "original_prior",
    "class_weighted",
    "os_smote",
    "os_ros",
    "os_kmeans_smote",
    "os_adasyn",
    "os_borderline_smote",
    "os_geometric_smote",
    "os_mgvae",
    "os_fol",
    "pste_smote",
    "pste_ros",
    "pste_kmeans_smote",
    "pste_adasyn",
    "pste_borderline_smote",
    "pste_geometric_smote",
    "pste_mgvae",
    "pste_fol",
    "brf",
    "balanced_bagging",
    "easy_ensemble",
    "rusboost",
)
METHOD_LABELS = {
    "original_prior": "Original-prior ensemble",
    "class_weighted": "Class-weighted ensemble",
    "os_smote": "SMOTE",
    "os_ros": "Random oversampling",
    "os_kmeans_smote": "KMeans-SMOTE",
    "os_adasyn": "ADASYN",
    "os_borderline_smote": "Borderline-SMOTE",
    "os_geometric_smote": "G-SMOTE",
    "os_mgvae": "MGVAE",
    "os_fol": "Fast Outward Ladder",
    "pste_smote": "PSTE--SMOTE",
    "pste_ros": "PSTE--ROS",
    "pste_kmeans_smote": "PSTE--KMeans-SMOTE",
    "pste_adasyn": "PSTE--ADASYN",
    "pste_borderline_smote": "PSTE--Borderline-SMOTE",
    "pste_geometric_smote": "PSTE--G-SMOTE",
    "pste_mgvae": "PSTE--MGVAE",
    "pste_fol": "PSTE--Fast Outward Ladder",
    "brf": "Balanced Random Forest",
    "balanced_bagging": "Balanced Bagging",
    "easy_ensemble": "EasyEnsemble",
    "rusboost": "RUSBoost",
}
MATCHED_PAIRS = (
    ("SMOTE", "pste_smote", "os_smote"),
    ("Random oversampling", "pste_ros", "os_ros"),
    ("KMeans-SMOTE", "pste_kmeans_smote", "os_kmeans_smote"),
    ("ADASYN", "pste_adasyn", "os_adasyn"),
    ("Borderline-SMOTE", "pste_borderline_smote", "os_borderline_smote"),
    ("G-SMOTE", "pste_geometric_smote", "os_geometric_smote"),
    ("MGVAE", "pste_mgvae", "os_mgvae"),
    ("Fast Outward Ladder", "pste_fol", "os_fol"),
)


def load_reference_results(
    path: str | Path = REFERENCE_METRICS,
) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def validate_reference(frame: pd.DataFrame) -> None:
    required = {
        "dataset",
        "screen_backbone",
        "seed",
        "fold",
        "logical_method",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"reference columns missing: {missing}")
    expected = 18 * 3 * 3 * 5 * 22
    if len(frame) != expected:
        raise AssertionError(f"reference rows: {len(frame):,} != {expected:,}")
    keys = [
        "dataset",
        "screen_backbone",
        "seed",
        "fold",
        "logical_method",
    ]
    if frame.duplicated(keys).any():
        raise AssertionError("duplicate reference fold/method keys")
    if frame.dataset.nunique() != 18:
        raise AssertionError("reference must contain 18 numerical tasks")
    if set(frame.screen_backbone) != set(BACKBONES):
        raise AssertionError("reference backbone set mismatch")
    if set(frame.logical_method) != set(METHODS):
        raise AssertionError("reference method set mismatch")
    if set(frame.seed.astype(int)) != set(SEEDS):
        raise AssertionError("reference seed set mismatch")
    if set(frame.fold.astype(int)) != set(FOLDS):
        raise AssertionError("reference fold set mismatch")
    counts = frame.groupby(
        ["dataset", "screen_backbone", "logical_method"]
    ).size()
    if not counts.eq(15).all():
        raise AssertionError("every task/backbone/method cell must have 15 rows")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise AssertionError("reference contains non-finite metrics")


def task_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    validate_reference(frame)
    return (
        frame.groupby(["dataset", "logical_method"], as_index=False)
        .pr_auc.mean()
        .pivot(index="dataset", columns="logical_method", values="pr_auc")
        .reindex(columns=METHODS)
        .sort_index()
    )


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for position, index in enumerate(order):
        running = max(
            running,
            min(1.0, (total - position) * values[index]),
        )
        adjusted[index] = running
    return adjusted


def exact_signed_rank(delta: Iterable[float]) -> tuple[float, float, int]:
    values = np.asarray(list(delta), dtype=float)
    values = values[values != 0]
    n = int(len(values))
    if n == 0:
        return 1.0, 0.0, 0
    ranks = rankdata(np.abs(values), method="average")
    scaled = np.rint(2.0 * ranks).astype(int)
    if not np.allclose(scaled, 2.0 * ranks, atol=1e-12, rtol=0.0):
        raise AssertionError("signed ranks could not be represented exactly")
    observed = int(scaled[values > 0].sum())
    total = int(scaled.sum())
    counts = [0] * (total + 1)
    counts[0] = 1
    reachable = 0
    for rank in scaled:
        for subtotal in range(reachable, -1, -1):
            if counts[subtotal]:
                counts[subtotal + rank] += counts[subtotal]
        reachable += int(rank)
    denominator = float(2**n)
    lower = sum(counts[: observed + 1]) / denominator
    upper = sum(counts[observed:]) / denominator
    p_value = min(1.0, 2.0 * min(lower, upper))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    effect = (positive - negative) / (positive + negative)
    return float(p_value), float(effect), n


def rank_summary(matrix: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ranks = matrix.rank(axis=1, method="average", ascending=False)
    statistic, p_value = friedmanchisquare(
        *[matrix[method].to_numpy(dtype=float) for method in METHODS]
    )
    ranking = pd.DataFrame(
        {
            "logical_method": METHODS,
            "method": [METHOD_LABELS[method] for method in METHODS],
            "mean_pr_auc": [
                float(matrix[method].mean()) for method in METHODS
            ],
            "average_rank": [
                float(ranks[method].mean()) for method in METHODS
            ],
        }
    ).sort_values(
        ["average_rank", "mean_pr_auc", "logical_method"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    summary = {
        "analysis": "numeric_task_18x22",
        "blocks": int(matrix.shape[0]),
        "methods": int(matrix.shape[1]),
        "degrees_of_freedom": int(matrix.shape[1] - 1),
        "friedman_chi_square": float(statistic),
        "friedman_p_value": float(p_value),
        "kendall_w": float(statistic)
        / (matrix.shape[0] * (matrix.shape[1] - 1)),
        "top_method": str(ranking.iloc[0].logical_method),
        "top_mean_pr_auc": float(ranking.iloc[0].mean_pr_auc),
        "top_average_rank": float(ranking.iloc[0].average_rank),
        "top_eight_all_pste": bool(
            ranking.head(8).logical_method.str.startswith("pste_").all()
        ),
    }
    return ranking.reset_index(drop=True), summary


def contrast(matrix: pd.DataFrame, method_a: str, method_b: str) -> dict:
    delta = (matrix[method_a] - matrix[method_b]).to_numpy(dtype=float)
    p_value, effect, nonzero = exact_signed_rank(delta)
    return {
        "method_a": method_a,
        "method_a_label": METHOD_LABELS.get(method_a, method_a),
        "method_b": method_b,
        "method_b_label": METHOD_LABELS.get(method_b, method_b),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "wins_a": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses_a": int(np.sum(delta < 0)),
        "rank_biserial": effect,
        "wilcoxon_nonzero_n": nonzero,
        "wilcoxon_exact_p": p_value,
    }


def focused_pairwise(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = [
        contrast(matrix, "pste_fol", method)
        for method in METHODS
        if method != "pste_fol"
    ]
    adjusted = holm_adjust(row["wilcoxon_exact_p"] for row in rows)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p_21"] = float(value)
    return pd.DataFrame(rows).sort_values(
        ["holm_p_21", "mean_delta"],
        ascending=[True, False],
        kind="mergesort",
    )


def all_pairwise(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_a, method_b in itertools.combinations(METHODS, 2):
        row = contrast(matrix, method_a, method_b)
        rows.append(
            {
                key: row[key]
                for key in (
                    "method_a",
                    "method_b",
                    "mean_delta",
                    "median_delta",
                    "rank_biserial",
                    "wilcoxon_nonzero_n",
                    "wilcoxon_exact_p",
                )
            }
        )
    adjusted = holm_adjust(row["wilcoxon_exact_p"] for row in rows)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p_231"] = float(value)
    if len(rows) != math.comb(22, 2):
        raise AssertionError("complete pairwise family must contain 231 rows")
    return pd.DataFrame(rows)


def matched_effects(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pste_methods = []
    standalone_methods = []
    for label, method_a, method_b in MATCHED_PAIRS:
        row = contrast(matrix, method_a, method_b)
        row["sampler"] = label
        rows.append(row)
        pste_methods.append(method_a)
        standalone_methods.append(method_b)
    adjusted = holm_adjust(row["wilcoxon_exact_p"] for row in rows)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p_8"] = float(value)

    family_delta = (
        matrix[pste_methods].mean(axis=1)
        - matrix[standalone_methods].mean(axis=1)
    ).to_numpy(dtype=float)
    p_value, effect, nonzero = exact_signed_rank(family_delta)
    rows.append(
        {
            "method_a": "pste_family_mean",
            "method_a_label": "PSTE family mean",
            "method_b": "standalone_family_mean",
            "method_b_label": "Standalone sampler family mean",
            "mean_delta": float(family_delta.mean()),
            "median_delta": float(np.median(family_delta)),
            "wins_a": int(np.sum(family_delta > 0)),
            "ties": int(np.sum(family_delta == 0)),
            "losses_a": int(np.sum(family_delta < 0)),
            "rank_biserial": effect,
            "wilcoxon_nonzero_n": nonzero,
            "wilcoxon_exact_p": p_value,
            "sampler": "Matched family mean",
            "holm_p_8": np.nan,
        }
    )
    return pd.DataFrame(rows)
