#!/usr/bin/env python3
"""Audit the packaged KBS primary and auxiliary reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.reference import (  # noqa: E402
    focused_pairwise,
    load_reference_results,
    matched_effects,
    rank_summary,
    task_matrix,
    validate_reference,
)


EXPECTED = {
    "rows": 17_820,
    "tasks": 18,
    "methods": 22,
    "backbones": 3,
    "top_mean_pr_auc": 0.723399719029844,
    "top_average_rank": 4.277777777777778,
    "friedman_chi_square": 204.59288537549423,
    "friedman_p_value": 4.512520352272491e-32,
    "kendall_w": 0.5412510195118895,
    "pste_fol_vs_fol_mean_delta": 0.01306177255379561,
    "pste_fol_vs_fol_wins": 15,
    "pste_fol_vs_fol_holm_p": 0.00295257568359375,
    "matched_family_mean_delta": 0.018949264768049357,
    "matched_family_wins": 18,
    "nested_rows": 810,
    "nested_mean_task_pr_auc": 0.720932005194895,
    "retained_minus_nested": 0.0024677138349491696,
    "retained_minus_nested_wins": 16,
    "retained_minus_nested_holm_p": 0.007720947265625,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if not np.isfinite(actual) or abs(float(actual) - float(expected)) > tol:
        raise AssertionError(
            f"{name}: actual={actual!r}, expected={expected!r}, tol={tol}"
        )


def audit_hashes(reference_root: Path) -> None:
    manifest_path = reference_root / "REFERENCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["artifacts"].items():
        path = reference_root / relative
        if not path.is_file():
            raise AssertionError(f"manifest artifact missing: {relative}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise AssertionError(
                f"{relative} SHA-256 mismatch: {actual} != {expected['sha256']}"
            )
        if path.stat().st_size != int(expected["bytes"]):
            raise AssertionError(f"{relative} byte count mismatch")


def audit_primary(reference_root: Path, tol: float) -> None:
    primary = reference_root / "primary"
    frame = load_reference_results(
        primary / "validated_fold_metrics_17820.csv"
    )
    validate_reference(frame)
    matrix = task_matrix(frame)
    ranking, summary = rank_summary(matrix)
    pairwise = focused_pairwise(matrix)
    matched = matched_effects(matrix)

    packaged_matrix = pd.read_csv(
        primary / "primary_numeric_task_prauc_18x22.csv",
        index_col=0,
        float_precision="round_trip",
    )
    pd.testing.assert_frame_equal(
        matrix,
        packaged_matrix,
        check_exact=False,
        atol=tol,
        rtol=0.0,
        check_names=False,
    )
    packaged_ranking = pd.read_csv(
        primary / "primary_numeric_task_ranking_22.csv",
        float_precision="round_trip",
    )
    pd.testing.assert_frame_equal(
        ranking,
        packaged_ranking,
        check_exact=False,
        atol=tol,
        rtol=0.0,
    )

    top = ranking.iloc[0]
    if top.logical_method != "pste_fol":
        raise AssertionError("PSTE--FOL is not the top packaged method")
    if not ranking.head(8).logical_method.str.startswith("pste_").all():
        raise AssertionError("the first eight ranks are not all PSTE variants")
    assert_close(
        "top mean AP",
        top.mean_pr_auc,
        EXPECTED["top_mean_pr_auc"],
        tol,
    )
    assert_close(
        "top average rank",
        top.average_rank,
        EXPECTED["top_average_rank"],
        tol,
    )
    for key in ("friedman_chi_square", "friedman_p_value", "kendall_w"):
        assert_close(key, summary[key], EXPECTED[key], max(tol, 1e-40))

    fol = pairwise.loc[pairwise.method_b.eq("os_fol")].iloc[0]
    assert_close(
        "PSTE--FOL vs FOL mean delta",
        fol.mean_delta,
        EXPECTED["pste_fol_vs_fol_mean_delta"],
        tol,
    )
    if int(fol.wins_a) != EXPECTED["pste_fol_vs_fol_wins"]:
        raise AssertionError("PSTE--FOL vs FOL win count mismatch")
    assert_close(
        "PSTE--FOL vs FOL Holm p",
        fol.holm_p_21,
        EXPECTED["pste_fol_vs_fol_holm_p"],
        tol,
    )

    family = matched.loc[matched.sampler.eq("Matched family mean")].iloc[0]
    assert_close(
        "matched PSTE family mean delta",
        family.mean_delta,
        EXPECTED["matched_family_mean_delta"],
        tol,
    )
    if int(family.wins_a) != EXPECTED["matched_family_wins"]:
        raise AssertionError("matched PSTE family win count mismatch")

    packaged_pairwise = pd.read_csv(
        primary / "primary_pste_fol_pairwise_holm_21.csv",
        float_precision="round_trip",
    )
    merge = pairwise.merge(
        packaged_pairwise,
        on=["method_a", "method_b"],
        suffixes=("_rebuilt", "_packaged"),
        validate="one_to_one",
    )
    for column in (
        "mean_delta",
        "median_delta",
        "rank_biserial",
        "wilcoxon_exact_p",
        "holm_p_21",
    ):
        difference = np.max(
            np.abs(
                merge[f"{column}_rebuilt"].to_numpy(dtype=float)
                - merge[f"{column}_packaged"].to_numpy(dtype=float)
            )
        )
        if difference > tol:
            raise AssertionError(
                f"rebuilt {column} differs from packaged by {difference}"
            )


def audit_nested(reference_root: Path, tol: float) -> None:
    auxiliary = reference_root / "auxiliary_nested_raw_vote"
    metrics = pd.read_csv(
        auxiliary / "validated_nested_fold_metrics_810.csv",
        float_precision="round_trip",
    )
    if len(metrics) != EXPECTED["nested_rows"]:
        raise AssertionError("nested-control row count mismatch")
    if metrics.dataset.nunique() != 18:
        raise AssertionError("nested control does not cover 18 tasks")
    contrasts = pd.read_csv(
        auxiliary / "nested_control_task_contrasts.csv",
        float_precision="round_trip",
    )
    retained = contrasts.loc[
        contrasts.contrast.eq(
            "retained_2to1_local_minus_nested_tuned_raw"
        )
    ].iloc[0]
    assert_close(
        "nested mean task AP",
        retained.second_mean_pr_auc,
        EXPECTED["nested_mean_task_pr_auc"],
        tol,
    )
    assert_close(
        "retained minus nested",
        retained.mean_delta_first_minus_second,
        EXPECTED["retained_minus_nested"],
        tol,
    )
    if int(retained.wins_first) != EXPECTED["retained_minus_nested_wins"]:
        raise AssertionError("retained-vs-nested win count mismatch")
    holm = pd.read_csv(
        auxiliary / "architecture_control_exact_wilcoxon_holm5.csv",
        float_precision="round_trip",
    )
    row = holm.loc[
        holm.contrast.eq(
            "retained_2to1_local_minus_nested_tuned_raw"
        )
    ].iloc[0]
    assert_close(
        "retained-vs-nested Holm p",
        row.holm_p_5,
        EXPECTED["retained_minus_nested_holm_p"],
        tol,
    )


def compare_generated(
    generated: Path,
    reference_root: Path,
    *,
    compare_values: bool,
    value_tolerance: float,
) -> None:
    frame = pd.read_csv(generated)
    validate_reference(frame)
    reference = load_reference_results(
        reference_root / "primary" / "validated_fold_metrics_17820.csv"
    )
    keys = [
        "dataset",
        "screen_backbone",
        "seed",
        "fold",
        "logical_method",
    ]
    expected = reference[keys].sort_values(keys).reset_index(drop=True)
    actual = frame[keys].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    if compare_values:
        expected_matrix = task_matrix(reference)
        actual_matrix = task_matrix(frame)
        worst = float(
            np.max(
                np.abs(
                    actual_matrix.to_numpy(dtype=float)
                    - expected_matrix.to_numpy(dtype=float)
                )
            )
        )
        if worst > value_tolerance:
            raise AssertionError(
                f"generated task means differ by up to {worst:.6g}, "
                f"above tolerance {value_tolerance}"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root",
        default=str(ROOT / "reference"),
    )
    parser.add_argument("--generated", default=None)
    parser.add_argument("--compare-generated-values", action="store_true")
    parser.add_argument("--generated-value-tol", type=float, default=1e-6)
    parser.add_argument("--tol", type=float, default=5e-12)
    args = parser.parse_args(argv)
    reference_root = Path(args.reference_root)

    audit_hashes(reference_root)
    audit_primary(reference_root, args.tol)
    audit_nested(reference_root, args.tol)
    if args.generated:
        compare_generated(
            Path(args.generated),
            reference_root,
            compare_values=args.compare_generated_values,
            value_tolerance=args.generated_value_tol,
        )
    print("reference audit OK")
    print("primary: 17,820 rows, 18 tasks, 3 backbones, 22 methods")
    print(
        "PSTE--FOL: mean AP 0.7233997190, average rank 4.2778; "
        "all eight PSTE variants occupy the first eight ranks"
    )
    print(
        "nested raw-vote control: mean AP 0.7209320052; "
        "retained PSTE--FOL +0.0024677138, 16/18 wins, Holm p 0.0077209473"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
