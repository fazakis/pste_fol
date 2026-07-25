#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.datasets import load_datasets
from pste_fol.experiment import (
    DEFAULT_CLASSIFIERS,
    DEFAULT_METHOD_GROUPS,
    DEFAULT_OVERSAMPLERS,
    run_experiment,
    write_results,
)
from pste_fol.provenance import (
    artifact_record,
    git_state,
    package_versions,
    runtime_environment,
    sha256_file,
)


def parse_json_dict(text: str | None) -> dict:
    if text is None or not str(text).strip():
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--oversampler-kwargs must be a JSON object")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the standalone fixed PSTE/Fast-Outward-Ladder experiment."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["paper"],
        help=(
            "Packaged names, paper/numeric18, bestangle25/all25, "
            "auxiliary/category7, or CSV paths."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data" / "bestangle25"),
        help="Directory containing packaged .joblib datasets.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column for CSV input; default is the final column.",
    )
    parser.add_argument(
        "--allow-categorical-encoding",
        action="store_true",
        help=(
            "Allow ordinal encoding of categorical CSV predictors for an "
            "explicitly auxiliary run. This is never enabled by paper mode."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 44, 49])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS)
    parser.add_argument("--oversamplers", nargs="+", default=DEFAULT_OVERSAMPLERS)
    parser.add_argument("--method-groups", nargs="+", default=DEFAULT_METHOD_GROUPS)
    parser.add_argument("--sampling-strategy", type=float, default=1.0)
    parser.add_argument(
        "--total-estimators",
        type=int,
        default=200,
        help="Total tree budget; PSTE divides it between both branches.",
    )
    parser.add_argument("--classifier-n-jobs", type=int, default=1)
    parser.add_argument(
        "--method-name-style",
        choices=["readable", "paper"],
        default="readable",
    )
    parser.add_argument(
        "--paper-exact",
        action="store_true",
        help=(
            "Run the locked KBS 18-task, 22-method, three-backbone primary "
            "protocol (17,820 rows with the default seeds/folds)."
        ),
    )
    parser.add_argument(
        "--oversampler-kwargs",
        default=None,
        help='JSON forwarded to oversamplers, e.g. \'{"max_candidates":4000}\'.',
    )

    # Exposed for custom studies. Paper modes reset every value to the fixed
    # retained protocol inside run_experiment().
    parser.add_argument("--pste-shadow-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--pste-correction-strength", type=float, default=0.15)
    parser.add_argument("--pste-local-prior-neighbors", type=int, default=31)
    parser.add_argument("--pste-local-prior-smoothing", type=float, default=10.0)
    parser.add_argument("--pste-local-shift-clip", type=float, default=2.0)
    parser.add_argument("--pste-support-neighbors", type=int, default=7)
    parser.add_argument("--pste-min-local-class-count", type=float, default=3.0)
    parser.add_argument(
        "--pste-blend-mode",
        choices=["prob", "logit"],
        default="prob",
    )
    parser.add_argument("--mgvae-epochs", type=int, default=200)
    parser.add_argument(
        "--mgvae-device",
        default="auto",
        help="PyTorch device for MGVAE: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--mgvae-torch-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "experiment_results.csv"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    paper_mode = "kbs" if args.paper_exact else None
    if args.paper_exact:
        args.datasets = ["paper"]
        args.data_dir = str(ROOT / "data" / "bestangle25")
        args.seeds = [42, 44, 49]
        args.folds = 5
        args.allow_categorical_encoding = False
    records = load_datasets(
        args.datasets,
        data_dir=args.data_dir,
        target=args.target,
        allow_categorical=bool(args.allow_categorical_encoding),
    )
    results = run_experiment(
        records,
        seeds=args.seeds,
        folds=args.folds,
        classifiers=args.classifiers,
        oversamplers=args.oversamplers,
        method_groups=args.method_groups,
        sampling_strategy=args.sampling_strategy,
        total_estimators=args.total_estimators,
        classifier_n_jobs=args.classifier_n_jobs,
        method_name_style=args.method_name_style,
        pste_shadow_fraction=args.pste_shadow_fraction,
        pste_correction_strength=args.pste_correction_strength,
        pste_local_prior_neighbors=args.pste_local_prior_neighbors,
        pste_local_prior_smoothing=args.pste_local_prior_smoothing,
        pste_local_shift_clip=args.pste_local_shift_clip,
        pste_support_neighbors=args.pste_support_neighbors,
        pste_min_local_class_count=args.pste_min_local_class_count,
        pste_blend_mode=args.pste_blend_mode,
        oversampler_kwargs=parse_json_dict(args.oversampler_kwargs),
        paper_mode=paper_mode,
        mgvae_epochs=args.mgvae_epochs,
        mgvae_device=args.mgvae_device,
        mgvae_torch_threads=args.mgvae_torch_threads,
    )
    output = write_results(results, args.output)
    manifest_path = Path(f"{output}.manifest.json")
    manifest = {
        "schema_version": 1,
        "runner": "scripts/run_experiment.py",
        "paper_exact": bool(args.paper_exact),
        "protocol": {
            "datasets": [record.name for record in records],
            "seeds": [int(value) for value in args.seeds],
            "folds": int(args.folds),
            "classifiers": list(
                DEFAULT_CLASSIFIERS if args.paper_exact else args.classifiers
            ),
            "oversamplers": list(
                DEFAULT_OVERSAMPLERS if args.paper_exact else args.oversamplers
            ),
            "method_groups": list(
                DEFAULT_METHOD_GROUPS if args.paper_exact else args.method_groups
            ),
            "sampling_strategy": (
                1.0 if args.paper_exact else float(args.sampling_strategy)
            ),
            "total_estimators": (
                200 if args.paper_exact else int(args.total_estimators)
            ),
            "mgvae_epochs": 200 if args.paper_exact else int(args.mgvae_epochs),
            "mgvae_device": str(args.mgvae_device),
            "mgvae_torch_threads": (
                1
                if args.paper_exact
                else int(args.mgvae_torch_threads)
            ),
        },
        "rows": int(len(results)),
        "logical_methods": sorted(results.logical_method.unique().tolist()),
        "environment": package_versions(),
        "runtime": runtime_environment(),
        "git": git_state(ROOT),
        "data_manifest_sha256": sha256_file(ROOT / "data" / "manifest.json"),
        "artifacts": {output.name: artifact_record(output)},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} rows to {output}")
    print(f"wrote run provenance to {manifest_path}")
    if len(results):
        print(
            results.groupby("method")["pr_auc"]
            .mean()
            .sort_values(ascending=False)
            .head(25)
            .to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
