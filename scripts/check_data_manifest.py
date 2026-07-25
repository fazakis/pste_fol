#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.datasets import (  # noqa: E402
    AUXILIARY_CATEGORY7_DATASETS,
    PRIMARY_NUMERIC18_DATASETS,
)

DATA_DIR = ROOT / "data" / "bestangle25"
MANIFEST = ROOT / "data" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def record_for(path: Path) -> dict:
    payload = joblib.load(path)
    X = np.asarray(payload["X"], dtype=float)
    y = np.asarray(payload["y"], dtype=int)
    values, counts = np.unique(y, return_counts=True)
    return {
        "name": str(payload.get("name", path.stem)),
        "file": str(path.relative_to(ROOT)),
        "source": str(payload.get("source", "packaged")),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
        "positive_class": "1 (minority)",
        "sha256": sha256(path),
        "metadata": dict(payload.get("metadata", {})),
    }


def current_manifest() -> dict:
    records = [record_for(path) for path in sorted(DATA_DIR.glob("*.joblib"))]
    return {
        "suite": "kbs_numeric18_primary_with_category7_auxiliary",
        "description": (
            "Packaged 25-dataset benchmark pool. The locked KBS primary "
            "analysis uses the 18 numerical tasks listed in primary_numeric18; "
            "seven category-bearing or otherwise auxiliary tasks are retained "
            "separately. Class 1 is the minority/positive class."
        ),
        "primary_numeric18": list(PRIMARY_NUMERIC18_DATASETS),
        "auxiliary_category7": list(AUXILIARY_CATEGORY7_DATASETS),
        "protocol": {
            "primary_datasets": 18,
            "packaged_datasets": 25,
            "methods": 22,
            "seeds": [42, 44, 49],
            "folds": 5,
            "backbones": ["rf", "extratrees", "bagged_cart"],
        },
        "datasets": records,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or regenerate the packaged bestangle25 data manifest."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite data/manifest.json from the packaged joblib files.",
    )
    args = parser.parse_args(argv)
    actual = current_manifest()
    if len(actual["datasets"]) != 25:
        raise AssertionError(
            f"expected 25 packaged datasets, found {len(actual['datasets'])}"
        )
    if args.write:
        MANIFEST.write_text(
            json.dumps(actual, indent=2, sort_keys=True, default=json_default) + "\n"
        )
        print(f"wrote {MANIFEST}")
        return 0
    expected = json.loads(MANIFEST.read_text())
    if actual != expected:
        raise AssertionError(
            "data/manifest.json does not match the packaged files; "
            "run this script with --write after intentional data changes"
        )
    print("data manifest OK: 25 files, shapes and SHA-256 hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
