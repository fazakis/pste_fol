from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import RIVAL_METHODS, canonical_classifier_name, make_classifier, make_rival_classifier
from .datasets import DatasetRecord, load_datasets
from .metrics import METRIC_NAMES, compute_metrics
from .oversampling import canonical_oversampler_name, make_oversampler
from .pste import PSTEClassifier
from .utils import positive_scores

DEFAULT_CLASSIFIERS = ["rf", "extratrees", "bagged_cart"]
DEFAULT_OVERSAMPLERS = [
    "random_over_sampler",
    "smote",
    "smote_tomek",
    "kmeans_smote",
    "adasyn",
    "borderline_smote",
    "deep_smote",
    "mgvae",
    "fast_outward_ladder",
]
DEFAULT_METHOD_GROUPS = ["native", "oversampler", "pste", "rivals"]
PAPER_EXACT_OVERSAMPLERS = [
    "smote_tomek",
    "smote",
    "random_over_sampler",
    "kmeans_smote",
    "adasyn",
    "borderline_smote",
    "mgvae",
    "deep_smote",
    "fast_outward_ladder",
]

RESULT_FIELDS = [
    "dataset",
    "dataset_source",
    "seed",
    "fold",
    "method",
    "paper_method_alias",
    "method_family",
    "classifier",
    "oversampler",
    "sampling_strategy",
    *METRIC_NAMES,
    "n_train_original",
    "n_test",
    "n_train_resampled",
    "n_generated",
    "runtime_seconds",
    "sampler_warning",
    "estimator_budget",
    "ensemble_components",
    "estimators_by_component",
    "notes",
]


def paper_sampler_token(oversampler: str) -> str:
    oversampler = canonical_oversampler_name(oversampler)
    if oversampler == "fast_outward_ladder":
        return "fast_outward_ladder_smote"
    return oversampler


def oversampler_method_name(classifier: str, oversampler: str, style: str = "pste") -> str:
    classifier = canonical_classifier_name(classifier)
    oversampler = canonical_oversampler_name(oversampler)
    token = "fast_outward_ladder" if oversampler == "fast_outward_ladder" else oversampler
    return f"{classifier}_{token}"


def pste_method_name(classifier: str, oversampler: str, style: str = "pste", fixed_mode: str | None = None) -> str:
    classifier = canonical_classifier_name(classifier)
    oversampler = canonical_oversampler_name(oversampler)
    suffix = f"_{fixed_mode}" if fixed_mode else ""
    if style == "paper":
        return f"fc_acc_ppob_{classifier}_{paper_sampler_token(oversampler)}{suffix}"
    token = "fast_outward_ladder" if oversampler == "fast_outward_ladder" else oversampler
    return f"pste_{classifier}_{token}{suffix}"


def pste_paper_alias(classifier: str, oversampler: str, fixed_mode: str | None = None) -> str:
    classifier = canonical_classifier_name(classifier)
    suffix = f"_{fixed_mode}" if fixed_mode else ""
    return f"fc_acc_ppob_{classifier}_{paper_sampler_token(oversampler)}{suffix}"


def preprocess_fold(X_train_raw, X_test_raw):
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    X_train = pipe.fit_transform(X_train_raw)
    X_test = pipe.transform(X_test_raw)
    return X_train, X_test


def fit_predict_native(classifier: str, X_train, y_train, X_test, seed: int, n_estimators: int, n_jobs: int, weighted: bool = False):
    clf = make_classifier(classifier, random_state=seed, n_estimators=n_estimators, n_jobs=n_jobs, weighted=weighted)
    clf.fit(X_train, y_train)
    y_score = positive_scores(clf, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score, len(y_train), 0, "", f"{classifier} {'weighted' if weighted else 'native'}", str(n_estimators), "class_weight=balanced" if weighted else ""


def fit_predict_oversampler(classifier: str, oversampler: str, X_train, y_train, X_test, seed: int, n_estimators: int, n_jobs: int, sampling_strategy: float, oversampler_kwargs: dict | None = None):
    oversampler = canonical_oversampler_name(oversampler)
    sampler = make_oversampler(oversampler, sampling_strategy=sampling_strategy, random_state=seed, **(oversampler_kwargs or {}))
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    clf = make_classifier(classifier, random_state=seed, n_estimators=n_estimators, n_jobs=n_jobs, weighted=False)
    clf.fit(X_res, y_res)
    y_score = positive_scores(clf, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    n_generated = int(getattr(sampler, "n_generated_", max(0, int(np.sum(y_res == 1) - np.sum(y_train == 1)))))
    notes = f"{classifier} trained after {oversampler} oversampling."
    return y_pred, y_score, len(y_res), n_generated, str(getattr(sampler, "warning_", "") or ""), oversampler, str(n_estimators), notes


def fit_predict_pste(
    classifier: str,
    oversampler: str,
    X_train,
    y_train,
    X_test,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    sampling_strategy: float,
    *,
    alpha_grid=None,
    modes=None,
    inner_cv_folds: int = 3,
    inner_cv_repeats: int = 2,
    default_alpha: float = 1.0 / 3.0,
    default_mode: str = "prob",
    prior_drift_max: float = 0.05,
    brier_degradation_max: float = 0.01,
    precision_topk_loss_max: float = 0.05,
    confidence_z: float = 1.96,
    inner_budget_cap: int = 80,
    oversampler_kwargs: dict | None = None,
):
    base = make_classifier(classifier, random_state=seed, n_estimators=n_estimators, n_jobs=n_jobs, weighted=False)
    clf = PSTEClassifier(
        base,
        oversampler=oversampler,
        oversampler_kwargs=oversampler_kwargs or {},
        sampling_strategy=sampling_strategy,
        total_estimators=n_estimators,
        alpha_grid=alpha_grid,
        modes=modes,
        default_alpha=default_alpha,
        default_mode=default_mode,
        prior_drift_max=prior_drift_max,
        brier_degradation_max=brier_degradation_max,
        precision_topk_loss_max=precision_topk_loss_max,
        inner_cv_folds=inner_cv_folds,
        inner_cv_repeats=inner_cv_repeats,
        confidence_z=confidence_z,
        inner_budget_cap=inner_budget_cap,
        random_state=seed,
        n_jobs=n_jobs,
    )
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    components = f"{classifier}_original:{clf.original_budget_}+{canonical_oversampler_name(oversampler)}_shadow:{clf.shadow_budget_}"
    notes = (
        f"PSTE selected mode={clf.selected_mode_}, shadow alpha={clf.selected_alpha_:.6f} "
        f"(approx original:shadow {clf.selected_ratio_}); selection_reason={clf.selection_reason_}; "
        f"inner_cv_splits={clf.inner_cv_splits_}; final refit on outer-training fold."
    )
    return y_pred, y_score, clf.n_resampled_, clf.n_generated_, clf.sampler_warning_, components, f"{clf.original_budget_}+{clf.shadow_budget_}", notes


def fit_predict_rival(method: str, X_train, y_train, X_test, seed: int, n_estimators: int, n_jobs: int):
    clf = make_rival_classifier(method, random_state=seed, n_estimators=n_estimators, n_jobs=n_jobs)
    clf.fit(X_train, y_train)
    y_score = positive_scores(clf, X_test)
    y_pred = clf.predict(X_test)
    return y_pred, y_score, len(y_train), 0, "", method, str(n_estimators), "external imbalanced-ensemble baseline"


def run_experiment(
    datasets: list[DatasetRecord] | list[str],
    *,
    seeds: list[int],
    folds: int,
    classifiers: list[str] | None = None,
    oversamplers: list[str] | None = None,
    method_groups: list[str] | None = None,
    sampling_strategy: float = 1.0,
    total_estimators: int = 200,
    classifier_n_jobs: int = 1,
    method_name_style: str = "pste",
    pste_alpha_grid=None,
    pste_modes=None,
    pste_inner_cv_folds: int = 3,
    pste_inner_cv_repeats: int = 2,
    pste_default_alpha: float = 1.0 / 3.0,
    pste_default_mode: str = "prob",
    pste_prior_drift_max: float = 0.05,
    pste_brier_degradation_max: float = 0.01,
    pste_precision_topk_loss_max: float = 0.05,
    pste_confidence_z: float = 1.96,
    pste_inner_budget_cap: int = 80,
    oversampler_kwargs: dict | None = None,
    paper_exact: bool = False,
) -> pd.DataFrame:
    if not datasets:
        raise ValueError("no datasets supplied")
    if isinstance(datasets[0], str):
        datasets = load_datasets(datasets)  # type: ignore[assignment]
    classifiers = [canonical_classifier_name(c) for c in (classifiers or DEFAULT_CLASSIFIERS)]
    oversamplers = [canonical_oversampler_name(o) for o in (oversamplers or DEFAULT_OVERSAMPLERS)]
    method_groups = [str(m).lower() for m in (method_groups or DEFAULT_METHOD_GROUPS)]
    oversampler_kwargs = dict(oversampler_kwargs or {})
    rows = []
    for record in datasets:  # type: ignore[assignment]
        y_all = np.asarray(record.y, dtype=int)
        min_count = int(np.min(np.bincount(y_all, minlength=2)))
        if min_count < folds:
            raise ValueError(f"dataset={record.name} has min class count {min_count}, too small for {folds}-fold CV")
        for seed in seeds:
            splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
            for fold, (train_idx, test_idx) in enumerate(splitter.split(record.X, record.y), start=1):
                X_train, X_test = preprocess_fold(record.X[train_idx], record.X[test_idx])
                y_train, y_test = y_all[train_idx], y_all[test_idx]
                jobs: list[tuple[str, str | None, str, str | None]] = []
                if "native" in method_groups:
                    for clf in classifiers:
                        jobs.append((clf, None, "native", None))
                if "weighted" in method_groups:
                    for clf in classifiers:
                        jobs.append((clf, None, "weighted", None))
                if "oversampler" in method_groups or "oversamplers" in method_groups:
                    for clf in classifiers:
                        for os_name in oversamplers:
                            jobs.append((clf, os_name, "oversampler", None))
                if "pste" in method_groups:
                    for clf in classifiers:
                        for os_name in oversamplers:
                            jobs.append((clf, os_name, "pste", None))
                        if paper_exact and "fast_outward_ladder" in oversamplers:
                            # The manuscript reference contains a fixed-logit PSTE--FOL
                            # variant in addition to adaptive-mode PSTE--FOL.
                            jobs.append((clf, "fast_outward_ladder", "pste", "logit"))
                if "rivals" in method_groups or "rival" in method_groups:
                    if paper_exact:
                        # The manuscript reference stores identical rival rows once per
                        # backbone result file. Repeating them here gives the same
                        # 27,000-row paper-exact shape.
                        for _clf_context in classifiers:
                            for rival in RIVAL_METHODS:
                                jobs.append((rival, None, "rival", None))
                    else:
                        for rival in RIVAL_METHODS:
                            jobs.append((rival, None, "rival", None))

                for clf, os_name, kind, fixed_mode in jobs:
                    started = time.perf_counter()
                    warning = ""
                    if kind == "native":
                        method = f"{clf}_none"
                        paper_alias = method
                        y_pred, y_score, n_resampled, n_generated, warning, components, estimators, notes = fit_predict_native(clf, X_train, y_train, X_test, int(seed), int(total_estimators), int(classifier_n_jobs), weighted=False)
                        family = "native_none"
                        oversampler_label = "none"
                        classifier_label = clf
                        sampling = "native"
                    elif kind == "weighted":
                        method = f"{clf}_class_weight_balanced"
                        paper_alias = method
                        y_pred, y_score, n_resampled, n_generated, warning, components, estimators, notes = fit_predict_native(clf, X_train, y_train, X_test, int(seed), int(total_estimators), int(classifier_n_jobs), weighted=True)
                        family = "weighted_or_cost_sensitive"
                        oversampler_label = "none"
                        classifier_label = clf
                        sampling = "native"
                    elif kind == "oversampler":
                        assert os_name is not None
                        method = oversampler_method_name(clf, os_name, style=method_name_style)
                        paper_alias = method
                        y_pred, y_score, n_resampled, n_generated, warning, components, estimators, notes = fit_predict_oversampler(clf, os_name, X_train, y_train, X_test, int(seed), int(total_estimators), int(classifier_n_jobs), float(sampling_strategy), oversampler_kwargs=oversampler_kwargs)
                        family = "proposed_oversampler_downstream" if os_name == "fast_outward_ladder" else "oversampler_downstream"
                        oversampler_label = os_name
                        classifier_label = clf
                        sampling = sampling_strategy
                    elif kind == "pste":
                        assert os_name is not None
                        method = pste_method_name(clf, os_name, style=method_name_style, fixed_mode=fixed_mode)
                        paper_alias = pste_paper_alias(clf, os_name, fixed_mode=fixed_mode)
                        y_pred, y_score, n_resampled, n_generated, warning, components, estimators, notes = fit_predict_pste(
                            clf,
                            os_name,
                            X_train,
                            y_train,
                            X_test,
                            int(seed),
                            int(total_estimators),
                            int(classifier_n_jobs),
                            float(sampling_strategy),
                            alpha_grid=pste_alpha_grid,
                            modes=[fixed_mode] if fixed_mode else pste_modes,
                            inner_cv_folds=pste_inner_cv_folds,
                            inner_cv_repeats=pste_inner_cv_repeats,
                            default_alpha=pste_default_alpha,
                            default_mode=fixed_mode or pste_default_mode,
                            prior_drift_max=pste_prior_drift_max,
                            brier_degradation_max=pste_brier_degradation_max,
                            precision_topk_loss_max=pste_precision_topk_loss_max,
                            confidence_z=pste_confidence_z,
                            inner_budget_cap=pste_inner_budget_cap,
                            oversampler_kwargs=oversampler_kwargs,
                        )
                        family = "proposed_pste"
                        oversampler_label = os_name
                        classifier_label = clf
                        sampling = sampling_strategy
                    else:
                        method = clf
                        paper_alias = clf
                        y_pred, y_score, n_resampled, n_generated, warning, components, estimators, notes = fit_predict_rival(clf, X_train, y_train, X_test, int(seed), int(total_estimators), int(classifier_n_jobs))
                        family = "rival_imbalanced_ensemble"
                        oversampler_label = "internal"
                        classifier_label = clf
                        sampling = "internal"

                    elapsed = round(float(time.perf_counter() - started), 6)
                    metrics = compute_metrics(y_test, y_pred, y_score)
                    rows.append(
                        {
                            "dataset": record.name,
                            "dataset_source": record.source,
                            "seed": int(seed),
                            "fold": int(fold),
                            "method": method,
                            "paper_method_alias": paper_alias,
                            "method_family": family,
                            "classifier": classifier_label,
                            "oversampler": oversampler_label,
                            "sampling_strategy": sampling,
                            **metrics,
                            "n_train_original": int(len(y_train)),
                            "n_test": int(len(y_test)),
                            "n_train_resampled": int(n_resampled),
                            "n_generated": int(n_generated),
                            "runtime_seconds": elapsed,
                            "sampler_warning": warning,
                            "estimator_budget": int(total_estimators),
                            "ensemble_components": components,
                            "estimators_by_component": estimators,
                            "notes": notes,
                        }
                    )
    return pd.DataFrame(rows, columns=RESULT_FIELDS)


def write_results(df: pd.DataFrame, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output
