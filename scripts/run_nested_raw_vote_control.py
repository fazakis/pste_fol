#!/usr/bin/env python3
"""Run the auxiliary nested-CV raw FOL soft-voting control.

This is deliberately not PSTE.  Each outer fold contains two equally sized
100-tree branches, one fitted on the original training fold and one on a FOL
shadow fold.  Three-fold inner CV selects only the raw probability-voting
weight.  Preprocessing, sampling, model fitting, and selection all remain
inside the outer-training fold.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.classifiers import canonical_classifier_name, make_classifier
from pste_fol.datasets import load_datasets
from pste_fol.experiment import preprocess_fold
from pste_fol.metrics import METRIC_NAMES, compute_metrics
from pste_fol.oversampling import make_oversampler
from pste_fol.provenance import (
    artifact_record,
    git_state,
    package_versions,
    runtime_environment,
    sha256_file,
)
from pste_fol.utils import positive_scores


BRANCH_ESTIMATORS = 100
DEFAULT_ALPHAS = tuple(
    round(float(value), 2) for value in np.arange(0.10, 0.901, 0.05)
)
DEFAULT_BACKBONES = ("rf", "extratrees", "bagged_cart")

KEY_FIELDS = ["dataset", "backbone", "seed", "fold"]
RESULT_FIELDS = [
    "dataset",
    "dataset_source",
    "backbone",
    "seed",
    "fold",
    "method",
    "selected_alpha",
    "selected_original_weight",
    "selected_shadow_weight",
    "candidate_alphas",
    "inner_pooled_ap_grid",
    "inner_mean_fold_ap_grid",
    "inner_best_pooled_ap",
    "inner_tie_count",
    "inner_folds",
    "original_estimators",
    "shadow_estimators",
    *METRIC_NAMES,
    "brier_score",
    "mean_probability",
    "true_prevalence",
    "inner_cv_runtime_seconds",
    "final_sampler_runtime_seconds",
    "final_branch_runtime_seconds",
    "final_scoring_runtime_seconds",
    "runtime_seconds",
    "n_train_original",
    "n_train_shadow",
    "n_generated",
    "pi_original",
    "pi_shadow",
    "sampler_warning",
]
PREDICTION_FIELDS = [
    "dataset",
    "dataset_source",
    "backbone",
    "seed",
    "fold",
    "sample_index",
    "y_true",
    "p_original",
    "p_shadow",
    "p_equal_raw",
    "p_selected_raw",
    "y_pred_selected",
    "selected_alpha",
]
ERROR_FIELDS = [*KEY_FIELDS, "error", "traceback"]


def _parse_alphas(values) -> tuple[float, ...]:
    alphas = tuple(sorted({round(float(value), 8) for value in values}))
    if not alphas or any(value < 0.0 or value > 1.0 for value in alphas):
        raise ValueError("candidate alphas must be a non-empty subset of [0, 1]")
    return alphas


def _blend(p_original, p_shadow, alpha: float) -> np.ndarray:
    return np.clip(
        (1.0 - float(alpha)) * np.asarray(p_original, dtype=float)
        + float(alpha) * np.asarray(p_shadow, dtype=float),
        0.0,
        1.0,
    )


def _fit_pair(
    backbone: str,
    X_train_raw,
    y_train,
    X_score_raw,
    seed: int,
    classifier_n_jobs: int,
    sampling_strategy: float,
):
    backbone = canonical_classifier_name(backbone)
    X_train, X_score = preprocess_fold(X_train_raw, X_score_raw)
    y_train = np.asarray(y_train, dtype=int)

    sampler_started = time.perf_counter()
    sampler = make_oversampler(
        "fast_outward_ladder",
        sampling_strategy=float(sampling_strategy),
        random_state=int(seed) + 202,
    )
    X_shadow, y_shadow = sampler.fit_resample(X_train, y_train)
    X_shadow = np.asarray(X_shadow, dtype=float)
    y_shadow = np.asarray(y_shadow, dtype=int)
    sampler_runtime = time.perf_counter() - sampler_started

    branch_started = time.perf_counter()
    original = make_classifier(
        backbone,
        random_state=int(seed) + 101,
        n_estimators=BRANCH_ESTIMATORS,
        n_jobs=int(classifier_n_jobs),
    )
    shadow = make_classifier(
        backbone,
        random_state=int(seed) + 303,
        n_estimators=BRANCH_ESTIMATORS,
        n_jobs=int(classifier_n_jobs),
    )
    original.fit(X_train, y_train)
    shadow.fit(X_shadow, y_shadow)
    p_original = np.asarray(positive_scores(original, X_score), dtype=float)
    p_shadow = np.asarray(positive_scores(shadow, X_score), dtype=float)
    branch_runtime = time.perf_counter() - branch_started

    metadata = {
        "sampler_runtime_seconds": float(sampler_runtime),
        "branch_runtime_seconds": float(branch_runtime),
        "n_train_original": int(len(y_train)),
        "n_train_shadow": int(len(y_shadow)),
        "n_generated": int(
            getattr(
                sampler,
                "n_generated_",
                max(0, int(np.sum(y_shadow == 1) - np.sum(y_train == 1))),
            )
        ),
        "pi_original": float(np.mean(y_train == 1)),
        "pi_shadow": float(np.mean(y_shadow == 1)),
        "sampler_warning": str(getattr(sampler, "warning_", "") or ""),
    }
    return p_original, p_shadow, metadata


def _select_alpha(
    backbone: str,
    X_outer_train_raw,
    y_outer_train,
    outer_seed: int,
    outer_fold: int,
    inner_folds: int,
    classifier_n_jobs: int,
    sampling_strategy: float,
    candidate_alphas: tuple[float, ...],
):
    y_outer_train = np.asarray(y_outer_train, dtype=int)
    n_splits = min(
        int(inner_folds),
        int(np.min(np.bincount(y_outer_train, minlength=2))),
    )
    if n_splits < 2:
        raise ValueError("too few minority examples for nested selection")

    started = time.perf_counter()
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(outer_seed) + 40000 + 7919 * int(outer_fold),
    )
    pooled_y = []
    pooled_original = []
    pooled_shadow = []
    fold_ap = {alpha: [] for alpha in candidate_alphas}

    for inner_fold, (sub_idx, val_idx) in enumerate(
        splitter.split(X_outer_train_raw, y_outer_train),
        start=1,
    ):
        inner_seed = (
            int(outer_seed)
            + 50000
            + 1009 * int(outer_fold)
            + 37 * int(inner_fold)
        )
        p_original, p_shadow, _ = _fit_pair(
            backbone,
            X_outer_train_raw[sub_idx],
            y_outer_train[sub_idx],
            X_outer_train_raw[val_idx],
            inner_seed,
            classifier_n_jobs,
            sampling_strategy,
        )
        y_val = y_outer_train[val_idx]
        pooled_y.append(np.asarray(y_val, dtype=int))
        pooled_original.append(p_original)
        pooled_shadow.append(p_shadow)
        for alpha in candidate_alphas:
            fold_ap[alpha].append(
                float(
                    average_precision_score(
                        y_val,
                        _blend(p_original, p_shadow, alpha),
                    )
                )
            )

    y_all = np.concatenate(pooled_y)
    p_original_all = np.concatenate(pooled_original)
    p_shadow_all = np.concatenate(pooled_shadow)
    pooled_grid = {
        alpha: float(
            average_precision_score(
                y_all,
                _blend(p_original_all, p_shadow_all, alpha),
            )
        )
        for alpha in candidate_alphas
    }
    mean_fold_grid = {
        alpha: float(np.mean(fold_ap[alpha])) for alpha in candidate_alphas
    }
    best = max(pooled_grid.values())
    tied = [
        alpha
        for alpha, value in pooled_grid.items()
        if np.isclose(value, best, rtol=0.0, atol=1e-12)
    ]
    selected = min(tied, key=lambda alpha: (abs(alpha - 0.5), alpha))
    return (
        float(selected),
        pooled_grid,
        mean_fold_grid,
        int(len(tied)),
        float(time.perf_counter() - started),
    )


def _grid_text(grid: dict[float, float]) -> str:
    return json.dumps(
        {f"{alpha:.2f}": round(float(value), 12) for alpha, value in grid.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


def _run_fold(job):
    (
        record,
        backbone,
        seed,
        fold,
        train_idx,
        test_idx,
        config,
    ) = job
    started = time.perf_counter()
    try:
        X_train_raw = record.X[train_idx]
        X_test_raw = record.X[test_idx]
        y_train = np.asarray(record.y[train_idx], dtype=int)
        y_test = np.asarray(record.y[test_idx], dtype=int)
        selected, pooled, mean_fold, ties, inner_runtime = _select_alpha(
            backbone,
            X_train_raw,
            y_train,
            int(seed),
            int(fold),
            int(config["inner_folds"]),
            int(config["classifier_n_jobs"]),
            float(config["sampling_strategy"]),
            tuple(config["candidate_alphas"]),
        )
        p_original, p_shadow, metadata = _fit_pair(
            backbone,
            X_train_raw,
            y_train,
            X_test_raw,
            int(seed),
            int(config["classifier_n_jobs"]),
            float(config["sampling_strategy"]),
        )
        scoring_started = time.perf_counter()
        p_equal = _blend(p_original, p_shadow, 0.5)
        p_selected = _blend(p_original, p_shadow, selected)
        y_pred = (p_selected >= 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred, p_selected)
        scoring_runtime = time.perf_counter() - scoring_started

        row = {
            "dataset": record.name,
            "dataset_source": record.source,
            "backbone": backbone,
            "seed": int(seed),
            "fold": int(fold),
            "method": "nested_tuned_raw_fol_equal100",
            "selected_alpha": selected,
            "selected_original_weight": 1.0 - selected,
            "selected_shadow_weight": selected,
            "candidate_alphas": ",".join(
                f"{alpha:.2f}" for alpha in config["candidate_alphas"]
            ),
            "inner_pooled_ap_grid": _grid_text(pooled),
            "inner_mean_fold_ap_grid": _grid_text(mean_fold),
            "inner_best_pooled_ap": max(pooled.values()),
            "inner_tie_count": ties,
            "inner_folds": int(config["inner_folds"]),
            "original_estimators": BRANCH_ESTIMATORS,
            "shadow_estimators": BRANCH_ESTIMATORS,
            **metrics,
            "brier_score": float(brier_score_loss(y_test, p_selected)),
            "mean_probability": float(np.mean(p_selected)),
            "true_prevalence": float(np.mean(y_test == 1)),
            "inner_cv_runtime_seconds": round(inner_runtime, 6),
            "final_sampler_runtime_seconds": round(
                metadata["sampler_runtime_seconds"], 6
            ),
            "final_branch_runtime_seconds": round(
                metadata["branch_runtime_seconds"], 6
            ),
            "final_scoring_runtime_seconds": round(scoring_runtime, 6),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "n_train_original": metadata["n_train_original"],
            "n_train_shadow": metadata["n_train_shadow"],
            "n_generated": metadata["n_generated"],
            "pi_original": metadata["pi_original"],
            "pi_shadow": metadata["pi_shadow"],
            "sampler_warning": metadata["sampler_warning"],
        }
        predictions = [
            {
                "dataset": record.name,
                "dataset_source": record.source,
                "backbone": backbone,
                "seed": int(seed),
                "fold": int(fold),
                "sample_index": int(sample_index),
                "y_true": int(label),
                "p_original": float(original),
                "p_shadow": float(shadow),
                "p_equal_raw": float(equal),
                "p_selected_raw": float(chosen),
                "y_pred_selected": int(prediction),
                "selected_alpha": selected,
            }
            for sample_index, label, original, shadow, equal, chosen, prediction in zip(
                np.asarray(test_idx, dtype=int),
                y_test,
                p_original,
                p_shadow,
                p_equal,
                p_selected,
                y_pred,
            )
        ]
        return row, predictions, None
    except Exception as exc:
        return (
            None,
            [],
            {
                "dataset": record.name,
                "backbone": backbone,
                "seed": int(seed),
                "fold": int(fold),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=12),
            },
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Nested-CV tuned raw 100/100 FOL soft-voting control"
    )
    parser.add_argument("--datasets", nargs="+", default=["paper"])
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "bestangle25"),
    )
    parser.add_argument(
        "--paper-exact",
        action="store_true",
        help=(
            "Lock the published 18-task, three-backbone, 3-inner/5-outer-fold "
            "protocol with seeds 42/44/49 and alpha grid 0.10:0.05:0.90."
        ),
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(DEFAULT_BACKBONES),
        choices=DEFAULT_BACKBONES,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 44, 49])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--candidate-alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument("--sampling-strategy", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--classifier-n-jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "nested_raw_vote"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory's three result files.",
    )
    return parser.parse_args(argv)


def _prepare_outputs(output_dir: Path, overwrite: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "nested_fold_metrics.csv",
        "predictions": output_dir / "nested_predictions.csv.gz",
        "errors": output_dir / "errors.csv",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"{names} already exist in {output_dir}; use --overwrite explicitly"
        )
    return paths


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.paper_exact:
        args.datasets = ["paper"]
        args.data_dir = str(ROOT / "data" / "bestangle25")
        args.backbones = list(DEFAULT_BACKBONES)
        args.seeds = [42, 44, 49]
        args.folds = 5
        args.inner_folds = 3
        args.candidate_alphas = list(DEFAULT_ALPHAS)
        args.sampling_strategy = 1.0
    if int(args.inner_folds) < 2:
        raise SystemExit("--inner-folds must be at least 2")
    alphas = _parse_alphas(args.candidate_alphas)
    records = load_datasets(args.datasets, data_dir=args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    paths = _prepare_outputs(output_dir, bool(args.overwrite))

    config = {
        "inner_folds": int(args.inner_folds),
        "candidate_alphas": alphas,
        "sampling_strategy": float(args.sampling_strategy),
        "classifier_n_jobs": int(args.classifier_n_jobs),
    }
    jobs = []
    for record in records:
        y = np.asarray(record.y, dtype=int)
        if int(np.min(np.bincount(y, minlength=2))) < int(args.folds):
            raise ValueError(
                f"{record.name}: class count is too small for {args.folds} folds"
            )
        for backbone in args.backbones:
            for seed in args.seeds:
                splitter = StratifiedKFold(
                    n_splits=int(args.folds),
                    shuffle=True,
                    random_state=int(seed),
                )
                for fold, (train_idx, test_idx) in enumerate(
                    splitter.split(record.X, record.y),
                    start=1,
                ):
                    jobs.append(
                        (
                            record,
                            backbone,
                            int(seed),
                            int(fold),
                            train_idx,
                            test_idx,
                            config,
                        )
                    )

    n_jobs = max(1, int(args.n_jobs))
    if n_jobs > 1:
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(variable, "1")

    with (
        paths["metrics"].open("w", newline="", encoding="utf-8") as metric_handle,
        gzip.open(
            paths["predictions"], "wt", newline="", encoding="utf-8"
        ) as prediction_handle,
        paths["errors"].open("w", newline="", encoding="utf-8") as error_handle,
    ):
        metric_writer = csv.DictWriter(metric_handle, fieldnames=RESULT_FIELDS)
        prediction_writer = csv.DictWriter(
            prediction_handle, fieldnames=PREDICTION_FIELDS
        )
        error_writer = csv.DictWriter(error_handle, fieldnames=ERROR_FIELDS)
        metric_writer.writeheader()
        prediction_writer.writeheader()
        error_writer.writeheader()

        if n_jobs == 1:
            iterator = map(_run_fold, jobs)
        else:
            pool = ProcessPoolExecutor(max_workers=n_jobs)
            iterator = pool.map(_run_fold, jobs, chunksize=1)
        try:
            for completed, (row, predictions, error) in enumerate(iterator, start=1):
                if row is not None:
                    metric_writer.writerow(row)
                    prediction_writer.writerows(predictions)
                if error is not None:
                    error_writer.writerow(error)
                if completed == 1 or completed % 10 == 0 or completed == len(jobs):
                    print(f"completed {completed}/{len(jobs)} outer folds", flush=True)
        finally:
            if n_jobs > 1:
                pool.shutdown()

    errors = pd.read_csv(paths["errors"])
    metrics = pd.read_csv(paths["metrics"])
    with gzip.open(
        paths["predictions"], "rt", encoding="utf-8"
    ) as prediction_input:
        prediction_rows = sum(1 for _ in prediction_input) - 1
    manifest = {
        "schema_version": 1,
        "runner": "scripts/run_nested_raw_vote_control.py",
        "paper_exact": bool(args.paper_exact),
        "protocol": {
            "datasets": [record.name for record in records],
            "backbones": list(args.backbones),
            "seeds": [int(value) for value in args.seeds],
            "outer_folds": int(args.folds),
            "inner_folds": int(args.inner_folds),
            "candidate_alphas": [float(value) for value in alphas],
            "sampling_strategy": float(args.sampling_strategy),
            "original_estimators": BRANCH_ESTIMATORS,
            "shadow_estimators": BRANCH_ESTIMATORS,
        },
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(prediction_rows),
        "error_rows": int(len(errors)),
        "environment": package_versions(),
        "runtime": runtime_environment(),
        "git": git_state(ROOT),
        "data_manifest_sha256": sha256_file(ROOT / "data" / "manifest.json"),
        "artifacts": {
            path.name: artifact_record(path) for path in paths.values()
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(metrics)} metric rows and {len(errors)} errors to {output_dir}"
    )
    return 1 if len(errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
