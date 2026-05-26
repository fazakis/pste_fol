#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pste_fol.datasets import load_datasets
from pste_fol.experiment import DEFAULT_CLASSIFIERS, DEFAULT_METHOD_GROUPS, DEFAULT_OVERSAMPLERS, PAPER_EXACT_OVERSAMPLERS, run_experiment, write_results


def parse_list(text: str | None):
    if text is None or str(text).strip() == "":
        return None
    return [p for p in str(text).replace(",", " ").split() if p]


def parse_json_dict(text: str | None) -> dict:
    if text is None or str(text).strip() == "":
        return {}
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--oversampler-kwargs must be a JSON object")
    return obj


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run standalone PSTE/Fast-Outward-Ladder reproducibility experiments.")
    p.add_argument("--datasets", nargs="+", default=["fast25"], help="Dataset names, 'fast25'/'paper'/'all', or CSV paths.")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "fast25"), help="Directory containing packaged .joblib datasets.")
    p.add_argument("--target", default=None, help="Target column for CSV datasets; default is the last column.")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 44, 49], help="CV seeds.")
    p.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds.")
    p.add_argument("--classifiers", nargs="+", default=DEFAULT_CLASSIFIERS, help="Tree classifiers: rf extratrees bagged_cart random_subspace_trees decision_tree.")
    p.add_argument("--oversamplers", nargs="+", default=DEFAULT_OVERSAMPLERS, help="Oversamplers: random_over_sampler smote smote_tomek kmeans_smote adasyn borderline_smote deep_smote mgvae fast_outward_ladder.")
    p.add_argument("--method-groups", nargs="+", default=DEFAULT_METHOD_GROUPS, help="Groups to run: native weighted oversampler pste rivals.")
    p.add_argument("--sampling-strategy", type=float, default=1.0, help="Minority/majority target ratio for oversampling.")
    p.add_argument("--total-estimators", type=int, default=200, help="Tree-estimator budget; PSTE splits this across original and shadow branches.")
    p.add_argument("--classifier-n-jobs", type=int, default=1, help="n_jobs passed to tree ensembles where supported.")
    p.add_argument("--method-name-style", choices=["pste", "paper"], default="pste", help="Use readable pste_* names or manuscript-compatible fc_acc_ppob_* names.")
    p.add_argument("--paper-exact", action="store_true", help="Use the manuscript method menu: 3 backbones, 9 oversamplers, adaptive PSTE variants, extra PSTE-FOL-logit variant, and rival rows repeated per backbone context. With default datasets/seeds/folds this emits 27,000 rows.")
    p.add_argument("--oversampler-kwargs", default=None, help="JSON object passed to oversampler constructors, e.g. '{\"max_candidates\":4000}'.")
    p.add_argument("--pste-alpha-grid", default=None, help="Comma/space alpha grid. Default: 0.00..1.00 by 0.02.")
    p.add_argument("--pste-modes", default=None, help="Comma/space blend modes. Default: prob,logit,priorcorr,priorcorr_logit.")
    p.add_argument("--pste-inner-cv-folds", type=int, default=3)
    p.add_argument("--pste-inner-cv-repeats", type=int, default=2)
    p.add_argument("--pste-default-alpha", type=float, default=1.0 / 3.0)
    p.add_argument("--pste-default-mode", default="prob")
    p.add_argument("--pste-prior-drift-max", type=float, default=0.05)
    p.add_argument("--pste-brier-degradation-max", type=float, default=0.01)
    p.add_argument("--pste-precision-topk-loss-max", type=float, default=0.05)
    p.add_argument("--pste-confidence-z", type=float, default=1.96)
    p.add_argument("--pste-inner-budget-cap", type=int, default=80)
    p.add_argument("--output", default=str(ROOT / "outputs" / "experiment_results.csv"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.paper_exact:
        args.classifiers = DEFAULT_CLASSIFIERS
        args.oversamplers = PAPER_EXACT_OVERSAMPLERS
        args.method_groups = DEFAULT_METHOD_GROUPS
        args.method_name_style = "paper"
        args.sampling_strategy = 1.0
        args.total_estimators = 200
        args.pste_inner_cv_folds = 3
        args.pste_inner_cv_repeats = 2
        args.pste_alpha_grid = None
        args.pste_modes = None
        args.pste_default_alpha = 1.0 / 3.0
        args.pste_default_mode = "prob"
        args.pste_prior_drift_max = 0.05
        args.pste_brier_degradation_max = 0.01
        args.pste_precision_topk_loss_max = 0.05
        args.pste_confidence_z = 1.96
        args.pste_inner_budget_cap = 80
    oversampler_kwargs = parse_json_dict(args.oversampler_kwargs)
    records = load_datasets(args.datasets, data_dir=args.data_dir, target=args.target)
    df = run_experiment(
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
        pste_alpha_grid=parse_list(args.pste_alpha_grid),
        pste_modes=parse_list(args.pste_modes),
        pste_inner_cv_folds=args.pste_inner_cv_folds,
        pste_inner_cv_repeats=args.pste_inner_cv_repeats,
        pste_default_alpha=args.pste_default_alpha,
        pste_default_mode=args.pste_default_mode,
        pste_prior_drift_max=args.pste_prior_drift_max,
        pste_brier_degradation_max=args.pste_brier_degradation_max,
        pste_precision_topk_loss_max=args.pste_precision_topk_loss_max,
        pste_confidence_z=args.pste_confidence_z,
        pste_inner_budget_cap=args.pste_inner_budget_cap,
        oversampler_kwargs=oversampler_kwargs,
        paper_exact=args.paper_exact,
    )
    out = write_results(df, args.output)
    print(f"wrote {len(df)} rows to {out}")
    if len(df):
        print(df.groupby("method")["pr_auc"].mean().sort_values(ascending=False).head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
