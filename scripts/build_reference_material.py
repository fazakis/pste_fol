#!/usr/bin/env python3
"""Rebuild the dependence-safe primary reference tables from fold metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.reference import (  # noqa: E402
    all_pairwise,
    focused_pairwise,
    load_reference_results,
    matched_effects,
    rank_summary,
    task_matrix,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default=str(
            ROOT
            / "reference"
            / "primary"
            / "validated_fold_metrics_17820.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "rebuilt_reference"),
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    matrix = task_matrix(load_reference_results(args.metrics))
    ranking, summary = rank_summary(matrix)
    matrix.to_csv(output / "primary_numeric_task_prauc_18x22.csv")
    ranking.to_csv(output / "primary_numeric_task_ranking_22.csv", index=False)
    focused_pairwise(matrix).to_csv(
        output / "primary_pste_fol_pairwise_holm_21.csv",
        index=False,
    )
    all_pairwise(matrix).to_csv(
        output / "primary_all_pairwise_holm_231.csv",
        index=False,
    )
    matched_effects(matrix).to_csv(
        output / "primary_matched_wrapper_effects.csv",
        index=False,
    )
    (output / "friedman_summaries.json").write_text(
        json.dumps({"numeric_task": summary}, indent=2) + "\n"
    )
    print(f"rebuilt primary reference material in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
