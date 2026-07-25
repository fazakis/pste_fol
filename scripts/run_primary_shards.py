#!/usr/bin/env python3
"""Resumable single-machine runner for the locked 18-task primary study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.datasets import (
    PRIMARY_NUMERIC18_DATASETS,
    load_dataset,
)
from pste_fol.experiment import run_experiment, write_results
from pste_fol.provenance import (
    artifact_record,
    git_state,
    package_versions,
    runtime_environment,
    sha256_file,
)
from pste_fol.reference import METHODS, validate_reference


SEEDS = [42, 44, 49]
FOLDS = 5
ROWS_PER_TASK = 3 * 5 * 3 * 22
KEY = ["dataset", "screen_backbone", "seed", "fold", "logical_method"]


def validate_shard(frame: pd.DataFrame, dataset: str) -> None:
    if len(frame) != ROWS_PER_TASK:
        raise AssertionError(
            f"{dataset}: shard rows {len(frame)} != {ROWS_PER_TASK}"
        )
    if set(frame.dataset) != {dataset}:
        raise AssertionError(f"{dataset}: shard dataset mismatch")
    if set(frame.seed.astype(int)) != set(SEEDS):
        raise AssertionError(f"{dataset}: shard seed mismatch")
    if set(frame.fold.astype(int)) != set(range(1, FOLDS + 1)):
        raise AssertionError(f"{dataset}: shard fold mismatch")
    if set(frame.screen_backbone) != {
        "bagged_cart",
        "randomforest",
        "extratrees",
    }:
        raise AssertionError(f"{dataset}: shard backbone mismatch")
    if set(frame.logical_method) != set(METHODS):
        raise AssertionError(f"{dataset}: shard method mismatch")
    if frame.duplicated(KEY).any():
        raise AssertionError(f"{dataset}: duplicate shard keys")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the locked primary benchmark one dataset at a time, keeping "
            "validated checkpoints that can be resumed on one machine."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(PRIMARY_NUMERIC18_DATASETS),
        help="A subset of the 18 primary task names; default is all 18.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "bestangle25"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "kbs_primary_sharded"),
    )
    parser.add_argument("--classifier-n-jobs", type=int, default=1)
    parser.add_argument("--mgvae-device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute selected task shards even if a valid checkpoint exists.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    datasets = list(dict.fromkeys(str(value) for value in args.datasets))
    unknown = sorted(set(datasets) - set(PRIMARY_NUMERIC18_DATASETS))
    if unknown:
        raise SystemExit(f"not in the locked numerical primary suite: {unknown}")

    output_dir = Path(args.output_dir).resolve()
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for position, dataset in enumerate(datasets, start=1):
        path = shard_dir / f"{dataset}.csv"
        if path.exists() and not args.overwrite:
            try:
                validate_shard(
                    pd.read_csv(path, float_precision="round_trip"),
                    dataset,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{path} is not a valid resumable checkpoint; inspect it "
                    "or rerun with --overwrite"
                ) from exc
            print(f"[{position}/{len(datasets)}] verified existing {dataset}")
            continue

        print(f"[{position}/{len(datasets)}] running {dataset}", flush=True)
        record = load_dataset(dataset, data_dir=args.data_dir)
        frame = run_experiment(
            [record],
            seeds=SEEDS,
            folds=FOLDS,
            paper_mode="kbs",
            classifier_n_jobs=int(args.classifier_n_jobs),
            mgvae_device=str(args.mgvae_device),
            mgvae_torch_threads=1,
        )
        validate_shard(frame, dataset)
        temporary = path.with_suffix(".csv.incomplete")
        write_results(frame.sort_values(KEY), temporary)
        os.replace(temporary, path)
        print(f"[{position}/{len(datasets)}] checkpointed {path}", flush=True)

    available = []
    missing = []
    for dataset in PRIMARY_NUMERIC18_DATASETS:
        path = shard_dir / f"{dataset}.csv"
        if not path.exists():
            missing.append(dataset)
            continue
        frame = pd.read_csv(path, float_precision="round_trip")
        validate_shard(frame, dataset)
        available.append(frame)

    if missing:
        print(
            "selected shards are complete; the full merged panel is waiting "
            f"for {len(missing)} task(s): {', '.join(missing)}"
        )
        return 0

    merged = pd.concat(available, ignore_index=True).sort_values(KEY)
    validate_reference(merged)
    merged_path = output_dir / "kbs_primary_17820.csv"
    temporary = merged_path.with_suffix(".csv.incomplete")
    write_results(merged, temporary)
    os.replace(temporary, merged_path)

    manifest = {
        "schema_version": 1,
        "runner": "scripts/run_primary_shards.py",
        "protocol": {
            "datasets": list(PRIMARY_NUMERIC18_DATASETS),
            "seeds": SEEDS,
            "folds": FOLDS,
            "backbones": ["rf", "extratrees", "bagged_cart"],
            "methods": 22,
            "total_estimators": 200,
            "mgvae_epochs": 200,
            "mgvae_device": str(args.mgvae_device),
            "mgvae_torch_threads": 1,
        },
        "rows": int(len(merged)),
        "environment": package_versions(),
        "runtime": runtime_environment(),
        "git": git_state(ROOT),
        "data_manifest_sha256": sha256_file(ROOT / "data" / "manifest.json"),
        "artifacts": {
            "kbs_primary_17820.csv": artifact_record(merged_path),
            "shards": {
                path.name: artifact_record(path)
                for path in sorted(shard_dir.glob("*.csv"))
            },
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"merged and validated {len(merged):,} rows in {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
