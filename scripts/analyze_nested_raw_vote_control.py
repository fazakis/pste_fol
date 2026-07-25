#!/usr/bin/env python3
"""Audit a standalone rerun of the auxiliary nested raw-vote control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.metrics import METRIC_NAMES, compute_metrics
from pste_fol.reference import exact_signed_rank, holm_adjust


REFERENCE = ROOT / "reference" / "auxiliary_nested_raw_vote"
KEY = ["dataset", "backbone", "seed", "fold"]
PREDICTION_KEY = [*KEY, "sample_index"]


class AuditError(RuntimeError):
    """Raised when a rerun cannot reproduce the locked control."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(values, *, seed: int, resamples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(int(resamples), len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def summarize(first, second, *, name: str, seed: int, resamples: int) -> dict:
    first, second = first.align(second, join="inner")
    if len(first) != 18 or not first.index.equals(second.index):
        raise AuditError(f"{name}: expected 18 aligned numerical tasks")
    delta = (first - second).to_numpy(dtype=float)
    p_value, effect, nonzero = exact_signed_rank(delta)
    low, high = bootstrap(delta, seed=seed, resamples=resamples)
    return {
        "contrast": name,
        "first_mean_pr_auc": float(first.mean()),
        "second_mean_pr_auc": float(second.mean()),
        "mean_delta_first_minus_second": float(delta.mean()),
        "median_delta_first_minus_second": float(np.median(delta)),
        "wins_first": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses_first": int(np.sum(delta < 0)),
        "rank_biserial": float(effect),
        "wilcoxon_nonzero_n": int(nonzero),
        "wilcoxon_exact_p": float(p_value),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit and analyze the nested raw-vote control rerun"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs" / "nested_raw_vote",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis destination; defaults to RUN_DIR/analysis.",
    )
    parser.add_argument("--reference-tolerance", type=float, default=1e-12)
    parser.add_argument("--resamples", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)

    metric_path = run_dir / "nested_fold_metrics.csv"
    prediction_path = run_dir / "nested_predictions.csv.gz"
    error_path = run_dir / "errors.csv"
    for path in (metric_path, prediction_path, error_path):
        if not path.exists():
            raise AuditError(f"missing rerun artifact: {path}")

    metrics = pd.read_csv(metric_path, float_precision="round_trip")
    predictions = pd.read_csv(
        prediction_path,
        compression="gzip",
        float_precision="round_trip",
    )
    errors = pd.read_csv(error_path)
    if len(errors):
        raise AuditError(f"rerun recorded {len(errors)} errors")
    if len(metrics) != 810:
        raise AuditError(f"metric rows {len(metrics)} != 810")
    if len(predictions) != 183_321:
        raise AuditError(f"prediction rows {len(predictions)} != 183,321")
    if metrics.duplicated(KEY).any():
        raise AuditError("duplicate outer-fold metric keys")
    if predictions.duplicated(PREDICTION_KEY).any():
        raise AuditError("duplicate held-out prediction keys")
    if metrics.dataset.nunique() != 18:
        raise AuditError("expected 18 numerical tasks")
    if set(metrics.backbone) != {"rf", "extratrees", "bagged_cart"}:
        raise AuditError("backbone coverage mismatch")
    if set(metrics.seed.astype(int)) != {42, 44, 49}:
        raise AuditError("seed coverage mismatch")
    if set(metrics.fold.astype(int)) != {1, 2, 3, 4, 5}:
        raise AuditError("fold coverage mismatch")
    if not (
        metrics.original_estimators.eq(100)
        & metrics.shadow_estimators.eq(100)
    ).all():
        raise AuditError("the deployment budget is not uniformly 100+100 trees")

    selection_rows = []
    for row in metrics.itertuples(index=False):
        grid = {
            float(alpha): float(score)
            for alpha, score in json.loads(row.inner_pooled_ap_grid).items()
        }
        best = max(grid.values())
        tied = [
            alpha
            for alpha, value in grid.items()
            if np.isclose(value, best, rtol=0.0, atol=1e-12)
        ]
        expected = min(tied, key=lambda alpha: (abs(alpha - 0.5), alpha))
        selection_rows.append(
            {
                **{column: getattr(row, column) for column in KEY},
                "reported_alpha": float(row.selected_alpha),
                "reselected_alpha": float(expected),
                "reported_best_inner_ap": float(row.inner_best_pooled_ap),
                "recomputed_best_inner_ap": float(best),
                "reported_tie_count": int(row.inner_tie_count),
                "recomputed_tie_count": int(len(tied)),
            }
        )
    selection = pd.DataFrame(selection_rows)
    if not np.allclose(
        selection.reported_alpha,
        selection.reselected_alpha,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AuditError("selected-alpha reconstruction mismatch")
    if not np.allclose(
        selection.reported_best_inner_ap,
        selection.recomputed_best_inner_ap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AuditError("inner AP-grid reconstruction mismatch")
    if not np.array_equal(
        selection.reported_tie_count,
        selection.recomputed_tie_count,
    ):
        raise AuditError("inner tie-count reconstruction mismatch")

    metric_rows = []
    for group_key, group in predictions.groupby(KEY, sort=False):
        alpha = group.selected_alpha.to_numpy(dtype=float)
        selected = (
            (1.0 - alpha) * group.p_original.to_numpy(dtype=float)
            + alpha * group.p_shadow.to_numpy(dtype=float)
        )
        if not np.allclose(
            selected,
            group.p_selected_raw,
            rtol=0.0,
            atol=1e-15,
        ):
            raise AuditError(f"raw-blend reconstruction mismatch at {group_key}")
        y_true = group.y_true.to_numpy(dtype=int)
        y_score = group.p_selected_raw.to_numpy(dtype=float)
        y_pred = (y_score >= 0.5).astype(int)
        recomputed = compute_metrics(y_true, y_pred, y_score)
        metric_rows.append(
            {
                **dict(zip(KEY, group_key)),
                **{f"recomputed_{key}": value for key, value in recomputed.items()},
                "recomputed_brier_score": float(
                    brier_score_loss(y_true, y_score)
                ),
            }
        )
    metric_audit = metrics.merge(
        pd.DataFrame(metric_rows),
        on=KEY,
        validate="one_to_one",
    )
    for metric in (*METRIC_NAMES, "brier_score"):
        if not np.allclose(
            metric_audit[metric],
            metric_audit[f"recomputed_{metric}"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AuditError(f"{metric} reconstruction mismatch")

    reference_metrics = pd.read_csv(
        REFERENCE / "validated_nested_fold_metrics_810.csv",
        float_precision="round_trip",
    )
    reference_predictions = pd.read_csv(
        REFERENCE / "validated_nested_predictions.csv.gz",
        compression="gzip",
        float_precision="round_trip",
    )
    metric_match = metrics.merge(
        reference_metrics,
        on=KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_rerun", "_reference"),
    )
    prediction_match = predictions.merge(
        reference_predictions,
        on=PREDICTION_KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_rerun", "_reference"),
    )
    if not metric_match._merge.eq("both").all():
        raise AuditError("reference metric-key coverage mismatch")
    if not prediction_match._merge.eq("both").all():
        raise AuditError("reference prediction-key coverage mismatch")
    if not np.array_equal(
        prediction_match.y_true_rerun,
        prediction_match.y_true_reference,
    ):
        raise AuditError("reference held-out labels differ")
    score_columns = (
        "p_original",
        "p_shadow",
        "p_equal_raw",
        "p_selected_raw",
    )
    score_differences = {
        column: float(
            np.max(
                np.abs(
                    prediction_match[f"{column}_rerun"]
                    - prediction_match[f"{column}_reference"]
                )
            )
        )
        for column in score_columns
    }
    max_reference_difference = max(score_differences.values())
    if max_reference_difference > float(args.reference_tolerance):
        raise AuditError(
            "reference predictions differ by "
            f"{max_reference_difference:.3g}, tolerance "
            f"{float(args.reference_tolerance):.3g}"
        )

    packaged_task = pd.read_csv(
        REFERENCE / "task_pr_auc_18x4.csv",
        index_col="dataset",
        float_precision="round_trip",
    ).sort_index()
    nested_task = metrics.groupby("dataset").pr_auc.mean().sort_index()
    max_task_difference = float(
        np.max(
            np.abs(
                nested_task.to_numpy(dtype=float)
                - packaged_task.nested_tuned_raw.to_numpy(dtype=float)
            )
        )
    )
    if max_task_difference > float(args.reference_tolerance):
        raise AuditError(
            f"nested task means differ from reference by {max_task_difference:.3g}"
        )

    contrasts = pd.DataFrame(
        [
            summarize(
                packaged_task["2to1_local"],
                nested_task,
                name="retained_2to1_local_minus_nested_tuned_raw",
                seed=2026072501,
                resamples=args.resamples,
            ),
            summarize(
                nested_task,
                packaged_task["1to1_raw"],
                name="nested_tuned_raw_minus_fixed_1to1_raw",
                seed=2026072502,
                resamples=args.resamples,
            ),
            summarize(
                nested_task,
                packaged_task["2to1_raw"],
                name="nested_tuned_raw_minus_fixed_2to1_raw",
                seed=2026072503,
                resamples=args.resamples,
            ),
        ]
    )
    packaged_family = pd.read_csv(
        REFERENCE / "architecture_control_exact_wilcoxon_holm5.csv",
        float_precision="round_trip",
    )
    family = pd.concat(
        [
            packaged_family.iloc[:4].drop(columns=["holm_p_5"]),
            contrasts.iloc[[0]],
        ],
        ignore_index=True,
    )
    family["holm_p_5"] = holm_adjust(family.wilcoxon_exact_p)

    task_output = packaged_task.copy()
    task_output["nested_tuned_raw"] = nested_task
    task_output = task_output[
        ["nested_tuned_raw", "1to1_raw", "2to1_raw", "2to1_local"]
    ]
    task_output.to_csv(output / "task_pr_auc_18x4.csv")
    metrics.sort_values(KEY).to_csv(
        output / "validated_nested_fold_metrics_810.csv",
        index=False,
    )
    predictions.sort_values(PREDICTION_KEY).to_csv(
        output / "validated_nested_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    selection.sort_values(KEY).to_csv(
        output / "inner_selection_reconstruction_audit_810.csv",
        index=False,
    )
    metric_audit.sort_values(KEY).to_csv(
        output / "prediction_metric_reconstruction_audit_810.csv",
        index=False,
    )
    contrasts.to_csv(output / "nested_control_task_contrasts.csv", index=False)
    family.to_csv(
        output / "architecture_control_exact_wilcoxon_holm5.csv",
        index=False,
    )
    alpha_counts = (
        metrics.groupby(["backbone", "selected_alpha"])
        .size()
        .rename("count")
        .reset_index()
    )
    overall_counts = (
        metrics.groupby("selected_alpha")
        .size()
        .rename("count")
        .reset_index()
        .assign(backbone="All")
    )
    pd.concat([overall_counts, alpha_counts], ignore_index=True).to_csv(
        output / "selected_alpha_counts.csv",
        index=False,
    )

    retained = family.loc[
        family.contrast.eq("retained_2to1_local_minus_nested_tuned_raw")
    ].iloc[0]
    summary = {
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "datasets": int(metrics.dataset.nunique()),
        "backbones": sorted(metrics.backbone.unique().tolist()),
        "max_reference_prediction_abs_difference": max_reference_difference,
        "reference_score_differences": score_differences,
        "max_reference_task_mean_abs_difference": max_task_difference,
        "nested_mean_task_pr_auc": float(nested_task.mean()),
        "selected_alpha_median": float(metrics.selected_alpha.median()),
        "selected_alpha_mode": float(
            metrics.selected_alpha.value_counts().index[0]
        ),
        "retained_minus_nested": retained.to_dict(),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# Nested-CV raw soft-voting control

- Scope: 18 numerical tasks, three backbones, seeds 42/44/49, five outer folds.
- Selection: three-fold inner CV, raw shadow weights 0.10 to 0.90 by 0.05.
- Deployment budget: 100 original-prior trees plus 100 FOL-shadow trees.
- Complete outer rows: {len(metrics)}
- Complete prediction rows: {len(predictions)}
- Maximum reference prediction difference: {max_reference_difference:.3g}

## Result

- Nested tuned raw mean task AP: {nested_task.mean():.6f}
- Retained PSTE--FOL minus nested tuned raw: {retained.mean_delta_first_minus_second:+.6f}
- Median difference: {retained.median_delta_first_minus_second:+.6f}
- W/T/L for retained: {int(retained.wins_first)}/{int(retained.ties)}/{int(retained.losses_first)}
- Rank-biserial correlation: {retained.rank_biserial:.6f}
- Task bootstrap 95% CI: [{retained.bootstrap_95_low:+.6f}, {retained.bootstrap_95_high:+.6f}]
- Exact Wilcoxon p: {retained.wilcoxon_exact_p:.8g}
- Holm p in the five-test RQ2 family: {retained.holm_p_5:.8g}
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(artifacts).to_csv(
        output / "artifact_manifest.csv",
        index=False,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
