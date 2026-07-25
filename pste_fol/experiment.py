from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classifiers import (
    RIVAL_METHODS,
    canonical_classifier_name,
    make_classifier,
    make_rival_classifier,
)
from .datasets import (
    AUXILIARY_CATEGORY7_DATASETS,
    DatasetRecord,
    load_datasets,
)
from .metrics import METRIC_NAMES, compute_metrics
from .oversampling import canonical_oversampler_name, make_oversampler
from .pste import PSTEClassifier
from .utils import positive_scores

DEFAULT_CLASSIFIERS = ["rf", "extratrees", "bagged_cart"]

# The locked KBS primary panel uses eight matched sampler families.
PAPER_RETAINED_OVERSAMPLERS = [
    "random_over_sampler",
    "smote",
    "kmeans_smote",
    "adasyn",
    "borderline_smote",
    "geometric_smote",
    "mgvae",
    "fast_outward_ladder",
]

DEFAULT_OVERSAMPLERS = list(PAPER_RETAINED_OVERSAMPLERS)
DEFAULT_METHOD_GROUPS = [
    "native",
    "weighted",
    "oversampler",
    "pste",
    "rivals",
]

RESULT_FIELDS = [
    "dataset",
    "dataset_source",
    "seed",
    "fold",
    "logical_method",
    "method",
    "paper_method_alias",
    "method_family",
    "classifier",
    "screen_backbone",
    "backbone_context",
    "oversampler",
    "sampling_strategy",
    *METRIC_NAMES,
    "n_train_original",
    "n_test",
    "n_train_resampled",
    "n_generated",
    "runtime_seconds",
    "shared_shadow_runtime_seconds",
    "mgvae_epochs",
    "sampler_warning",
    "estimator_budget",
    "ensemble_components",
    "estimators_by_component",
    "notes",
]


SCREEN_BACKBONES = {
    "rf": "randomforest",
    "extratrees": "extratrees",
    "bagged_cart": "bagged_cart",
}

LOGICAL_SAMPLERS = {
    "random_over_sampler": "ros",
    "smote": "smote",
    "kmeans_smote": "kmeans_smote",
    "adasyn": "adasyn",
    "borderline_smote": "borderline_smote",
    "geometric_smote": "geometric_smote",
    "mgvae": "mgvae",
    "fast_outward_ladder": "fol",
}


def logical_method_name(kind: str, token: str | None = None) -> str:
    if kind == "native":
        return "original_prior"
    if kind == "weighted":
        return "class_weighted"
    if kind == "rival":
        mapping = {
            "balanced_random_forest": "brf",
            "balanced_bagging": "balanced_bagging",
            "easy_ensemble": "easy_ensemble",
            "rus_boost": "rusboost",
        }
        return mapping[str(token)]
    sampler = LOGICAL_SAMPLERS[str(token)]
    if kind == "oversampler":
        return f"os_{sampler}"
    if kind == "pste":
        return f"pste_{sampler}"
    raise ValueError(f"unknown method kind {kind!r}")


def paper_sampler_token(oversampler: str) -> str:
    oversampler = canonical_oversampler_name(oversampler)
    if oversampler == "fast_outward_ladder":
        return "fast_outward_ladder_smote"
    return oversampler


def oversampler_method_name(classifier: str, oversampler: str) -> str:
    classifier = canonical_classifier_name(classifier)
    oversampler = canonical_oversampler_name(oversampler)
    token = "fast_outward_ladder" if oversampler == "fast_outward_ladder" else oversampler
    return f"{classifier}_{token}"


def pste_method_name(
    classifier: str,
    oversampler: str,
    *,
    style: str = "readable",
    blend_mode: str = "prob",
) -> str:
    classifier = canonical_classifier_name(classifier)
    oversampler = canonical_oversampler_name(oversampler)
    mode_suffix = "_logit" if blend_mode == "logit" else ""
    if style == "paper":
        return (
            f"ppob_{classifier}_{paper_sampler_token(oversampler)}"
            f"_2to1_controlcorr015_conf{mode_suffix}"
        )
    token = "fol" if oversampler == "fast_outward_ladder" else oversampler
    return f"pste_{classifier}_{token}{mode_suffix}"


def preprocess_fold(X_train_raw, X_test_raw):
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_train = preprocess.fit_transform(X_train_raw)
    X_test = preprocess.transform(X_test_raw)
    return X_train, X_test


def fit_predict_native(
    classifier: str,
    X_train,
    y_train,
    X_test,
    seed: int,
    n_estimators: int,
    n_jobs: int,
    *,
    weighted: bool = False,
):
    classifier = canonical_classifier_name(classifier)
    estimator = make_classifier(
        classifier,
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        weighted=weighted,
    )
    estimator.fit(X_train, y_train)
    y_score = positive_scores(estimator, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    notes = (
        f"{classifier} class_weight=balanced"
        if weighted
        else f"{classifier} trained on the original-prior fold"
    )
    return (
        y_pred,
        y_score,
        len(y_train),
        0,
        "",
        f"{classifier}_{'weighted' if weighted else 'none'}",
        str(n_estimators),
        notes,
    )


def fit_predict_oversampler(
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
    oversampler_kwargs: dict | None = None,
    precomputed_shadow: tuple[np.ndarray, np.ndarray, dict] | None = None,
):
    classifier = canonical_classifier_name(classifier)
    oversampler = canonical_oversampler_name(oversampler)
    if precomputed_shadow is None:
        sampler = make_oversampler(
            oversampler,
            sampling_strategy=sampling_strategy,
            random_state=seed,
            **(oversampler_kwargs or {}),
        )
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        warning = str(getattr(sampler, "warning_", "") or "")
        n_generated = int(
            getattr(
                sampler,
                "n_generated_",
                max(0, int(np.sum(y_resampled == 1) - np.sum(y_train == 1))),
            )
        )
    else:
        X_resampled, y_resampled, metadata = precomputed_shadow
        sampler = None
        warning = str(metadata.get("warning", "") or "")
        n_generated = int(metadata["n_generated"])
    estimator = make_classifier(
        classifier,
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        weighted=False,
    )
    estimator.fit(X_resampled, y_resampled)
    y_score = positive_scores(estimator, X_test)
    y_pred = (y_score >= 0.5).astype(int)
    return (
        y_pred,
        y_score,
        len(y_resampled),
        n_generated,
        warning,
        paper_sampler_token(oversampler),
        str(n_estimators),
        f"{classifier} trained after {oversampler} oversampling",
    )


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
    shadow_fraction: float = 1.0 / 3.0,
    correction_strength: float = 0.15,
    local_prior_neighbors: int = 31,
    local_prior_smoothing: float = 10.0,
    local_shift_clip: float = 2.0,
    support_neighbors: int = 7,
    min_local_class_count: float = 3.0,
    blend_mode: str = "prob",
    oversampler_kwargs: dict | None = None,
    precomputed_shadow: tuple[np.ndarray, np.ndarray, dict] | None = None,
):
    classifier = canonical_classifier_name(classifier)
    oversampler = canonical_oversampler_name(oversampler)
    base = make_classifier(
        classifier,
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        weighted=False,
    )
    estimator = PSTEClassifier(
        base,
        oversampler=oversampler,
        oversampler_kwargs=oversampler_kwargs or {},
        sampling_strategy=sampling_strategy,
        total_estimators=n_estimators,
        shadow_fraction=shadow_fraction,
        correction_strength=correction_strength,
        local_prior_neighbors=local_prior_neighbors,
        local_prior_smoothing=local_prior_smoothing,
        local_shift_clip=local_shift_clip,
        support_neighbors=support_neighbors,
        min_local_class_count=min_local_class_count,
        blend_mode=blend_mode,
        random_state=seed,
        n_jobs=n_jobs,
    )
    if precomputed_shadow is None:
        estimator.fit(X_train, y_train)
    else:
        X_shadow, y_shadow, metadata = precomputed_shadow
        estimator.fit_with_shadow(
            X_train,
            y_train,
            X_shadow,
            y_shadow,
            shadow_support=np.asarray(X_shadow, dtype=float)[
                np.asarray(y_shadow, dtype=int) == 1
            ],
            n_generated=int(metadata["n_generated"]),
            sampler_warning=str(metadata.get("warning", "") or ""),
        )
    y_score = estimator.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    components = (
        f"{classifier}_original:{estimator.original_budget_}"
        f"+{paper_sampler_token(oversampler)}_shadow:{estimator.shadow_budget_}"
    )
    notes = (
        f"Fixed confidence-gated PSTE {blend_mode} blend; "
        f"original:shadow score weight 2:1; estimator budget {n_estimators}; "
        f"gamma={correction_strength:g}; local_prior_k={local_prior_neighbors}; "
        f"local_prior_smooth={local_prior_smoothing:g}; "
        f"local_shift_clip={local_shift_clip:g}; support_gate_k={support_neighbors}; "
        f"min_local_class_count={min_local_class_count:g}; no inner validation"
    )
    return (
        y_pred,
        y_score,
        estimator.n_resampled_,
        estimator.n_generated_,
        estimator.sampler_warning_,
        components,
        f"{estimator.original_budget_}+{estimator.shadow_budget_}",
        notes,
    )


def fit_predict_rival(
    method: str,
    X_train,
    y_train,
    X_test,
    seed: int,
    n_estimators: int,
    n_jobs: int,
):
    estimator = make_rival_classifier(
        method,
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    estimator.fit(X_train, y_train)
    y_score = positive_scores(estimator, X_test)
    y_pred = estimator.predict(X_test)
    return (
        y_pred,
        y_score,
        len(y_train),
        0,
        "",
        method,
        str(n_estimators),
        "external imbalanced-ensemble baseline",
    )


def build_shared_mgvae_shadow(
    X_train,
    y_train,
    *,
    seed: int,
    sampling_strategy: float,
    epochs: int,
    device: str,
    torch_threads: int | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the one fold-level MGVAE shadow shared by both roles/backbones."""

    started = time.perf_counter()
    sampler = make_oversampler(
        "mgvae",
        sampling_strategy=float(sampling_strategy),
        random_state=int(seed),
        epochs=int(epochs),
        latent_dim=10,
        hidden_size=300,
        number_components=700,
        batch_size=100,
        lr=0.005,
        weight_decay=0.0,
        kld_weight=1.0,
        device=str(device),
        torch_threads=torch_threads,
    )
    X_shadow, y_shadow = sampler.fit_resample(X_train, y_train)
    metadata = {
        "n_generated": int(getattr(sampler, "n_generated_", 0)),
        "warning": str(getattr(sampler, "warning_", "") or ""),
        "training_loss": float(getattr(sampler, "training_loss_", np.nan)),
        "runtime_seconds": float(time.perf_counter() - started),
        "epochs": int(epochs),
        "sampler_seed": int(seed),
    }
    return (
        np.asarray(X_shadow, dtype=float),
        np.asarray(y_shadow, dtype=int),
        metadata,
    )


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
    method_name_style: str = "readable",
    pste_shadow_fraction: float = 1.0 / 3.0,
    pste_correction_strength: float = 0.15,
    pste_local_prior_neighbors: int = 31,
    pste_local_prior_smoothing: float = 10.0,
    pste_local_shift_clip: float = 2.0,
    pste_support_neighbors: int = 7,
    pste_min_local_class_count: float = 3.0,
    pste_blend_mode: str = "prob",
    oversampler_kwargs: dict | None = None,
    paper_mode: str | None = None,
    mgvae_epochs: int = 200,
    mgvae_device: str = "auto",
    mgvae_torch_threads: int | None = 1,
) -> pd.DataFrame:
    """Run the standalone benchmark on one machine.

    ``paper_mode="kbs"`` emits the locked 22-method primary panel in each of
    the three backbone contexts. The full design has 17,820 rows.
    """

    if not datasets:
        raise ValueError("no datasets supplied")
    if isinstance(datasets[0], str):
        datasets = load_datasets(datasets)  # type: ignore[assignment]

    if paper_mode is not None:
        paper_mode = str(paper_mode).lower()
        if paper_mode in {"retained", "paper", "current"}:
            paper_mode = "kbs"
        if paper_mode != "kbs":
            raise ValueError("paper_mode must be None or 'kbs'")
        classifiers = list(DEFAULT_CLASSIFIERS)
        oversamplers = list(PAPER_RETAINED_OVERSAMPLERS)
        method_groups = list(DEFAULT_METHOD_GROUPS)
        sampling_strategy = 1.0
        total_estimators = 200
        method_name_style = "paper"
        pste_shadow_fraction = 1.0 / 3.0
        pste_correction_strength = 0.15
        pste_local_prior_neighbors = 31
        pste_local_prior_smoothing = 10.0
        pste_local_shift_clip = 2.0
        pste_support_neighbors = 7
        pste_min_local_class_count = 3.0
        pste_blend_mode = "prob"
        oversampler_kwargs = {}
        mgvae_epochs = 200
        mgvae_torch_threads = 1

    classifiers = [
        canonical_classifier_name(classifier)
        for classifier in (classifiers or DEFAULT_CLASSIFIERS)
    ]
    oversamplers = [
        canonical_oversampler_name(oversampler)
        for oversampler in (oversamplers or DEFAULT_OVERSAMPLERS)
    ]
    method_groups = [
        str(group).lower() for group in (method_groups or DEFAULT_METHOD_GROUPS)
    ]
    oversampler_kwargs = dict(oversampler_kwargs or {})
    rows = []

    for record in datasets:  # type: ignore[assignment]
        if (
            paper_mode == "kbs"
            and str(record.name) in set(AUXILIARY_CATEGORY7_DATASETS)
        ):
            raise ValueError(
                f"{record.name} is category-bearing or auxiliary and is not "
                "part of the locked 18-task numerical primary panel"
            )
        y_all = np.asarray(record.y, dtype=int)
        min_count = int(np.min(np.bincount(y_all, minlength=2)))
        if min_count < folds:
            raise ValueError(
                f"dataset={record.name} has minority count {min_count}, "
                f"too small for {folds}-fold CV"
            )

        for seed in seeds:
            splitter = StratifiedKFold(
                n_splits=int(folds),
                shuffle=True,
                random_state=int(seed),
            )
            for fold, (train_indices, test_indices) in enumerate(
                splitter.split(record.X, record.y),
                start=1,
            ):
                X_train, X_test = preprocess_fold(
                    record.X[train_indices],
                    record.X[test_indices],
                )
                y_train = y_all[train_indices]
                y_test = y_all[test_indices]
                mgvae_shadow = None
                if "mgvae" in oversamplers and (
                    "oversampler" in method_groups
                    or "oversamplers" in method_groups
                    or "pste" in method_groups
                ):
                    mgvae_shadow = build_shared_mgvae_shadow(
                        X_train,
                        y_train,
                        seed=int(seed),
                        sampling_strategy=float(sampling_strategy),
                        epochs=int(mgvae_epochs),
                        device=str(mgvae_device),
                        torch_threads=mgvae_torch_threads,
                    )

                # classifier, oversampler, kind, blend mode, backbone context
                jobs: list[tuple[str, str | None, str, str, str]] = []
                if "native" in method_groups:
                    jobs.extend(
                        (classifier, None, "native", "prob", classifier)
                        for classifier in classifiers
                    )
                if "weighted" in method_groups:
                    jobs.extend(
                        (classifier, None, "weighted", "prob", classifier)
                        for classifier in classifiers
                    )
                if "oversampler" in method_groups or "oversamplers" in method_groups:
                    jobs.extend(
                        (classifier, oversampler, "oversampler", "prob", classifier)
                        for classifier in classifiers
                        for oversampler in oversamplers
                    )
                if "pste" in method_groups:
                    jobs.extend(
                        (
                            classifier,
                            oversampler,
                            "pste",
                            pste_blend_mode,
                            classifier,
                        )
                        for classifier in classifiers
                        for oversampler in oversamplers
                    )
                if "rivals" in method_groups or "rival" in method_groups:
                    contexts = classifiers if paper_mode is not None else ["shared"]
                    jobs.extend(
                        (rival, None, "rival", "prob", context)
                        for context in contexts
                        for rival in RIVAL_METHODS
                    )

                for classifier, oversampler, kind, blend_mode, context in jobs:
                    started = time.perf_counter()
                    if kind == "native":
                        method = f"{classifier}_none"
                        (
                            y_pred,
                            y_score,
                            n_resampled,
                            n_generated,
                            warning,
                            components,
                            estimators,
                            notes,
                        ) = fit_predict_native(
                            classifier,
                            X_train,
                            y_train,
                            X_test,
                            int(seed),
                            int(total_estimators),
                            int(classifier_n_jobs),
                        )
                        family = "native_none"
                        oversampler_label = "none"
                        classifier_label = classifier
                        sampling = "native"
                    elif kind == "weighted":
                        method = f"{classifier}_class_weight_balanced"
                        (
                            y_pred,
                            y_score,
                            n_resampled,
                            n_generated,
                            warning,
                            components,
                            estimators,
                            notes,
                        ) = fit_predict_native(
                            classifier,
                            X_train,
                            y_train,
                            X_test,
                            int(seed),
                            int(total_estimators),
                            int(classifier_n_jobs),
                            weighted=True,
                        )
                        family = "weighted_or_cost_sensitive"
                        oversampler_label = "none"
                        classifier_label = classifier
                        sampling = "native"
                    elif kind == "oversampler":
                        assert oversampler is not None
                        method = oversampler_method_name(classifier, oversampler)
                        (
                            y_pred,
                            y_score,
                            n_resampled,
                            n_generated,
                            warning,
                            components,
                            estimators,
                            notes,
                        ) = fit_predict_oversampler(
                            classifier,
                            oversampler,
                            X_train,
                            y_train,
                            X_test,
                            int(seed),
                            int(total_estimators),
                            int(classifier_n_jobs),
                            float(sampling_strategy),
                            oversampler_kwargs=oversampler_kwargs,
                            precomputed_shadow=(
                                mgvae_shadow if oversampler == "mgvae" else None
                            ),
                        )
                        family = (
                            "proposed_oversampler_downstream"
                            if oversampler == "fast_outward_ladder"
                            else "oversampler_downstream"
                        )
                        oversampler_label = oversampler
                        classifier_label = classifier
                        sampling = sampling_strategy
                    elif kind == "pste":
                        assert oversampler is not None
                        method = pste_method_name(
                            classifier,
                            oversampler,
                            style=method_name_style,
                            blend_mode=blend_mode,
                        )
                        (
                            y_pred,
                            y_score,
                            n_resampled,
                            n_generated,
                            warning,
                            components,
                            estimators,
                            notes,
                        ) = fit_predict_pste(
                            classifier,
                            oversampler,
                            X_train,
                            y_train,
                            X_test,
                            int(seed),
                            int(total_estimators),
                            int(classifier_n_jobs),
                            float(sampling_strategy),
                            shadow_fraction=pste_shadow_fraction,
                            correction_strength=pste_correction_strength,
                            local_prior_neighbors=pste_local_prior_neighbors,
                            local_prior_smoothing=pste_local_prior_smoothing,
                            local_shift_clip=pste_local_shift_clip,
                            support_neighbors=pste_support_neighbors,
                            min_local_class_count=pste_min_local_class_count,
                            blend_mode=blend_mode,
                            oversampler_kwargs=oversampler_kwargs,
                            precomputed_shadow=(
                                mgvae_shadow if oversampler == "mgvae" else None
                            ),
                        )
                        family = "proposed_ppob"
                        oversampler_label = oversampler
                        classifier_label = classifier
                        sampling = sampling_strategy
                    else:
                        method = classifier
                        (
                            y_pred,
                            y_score,
                            n_resampled,
                            n_generated,
                            warning,
                            components,
                            estimators,
                            notes,
                        ) = fit_predict_rival(
                            classifier,
                            X_train,
                            y_train,
                            X_test,
                            int(seed),
                            int(total_estimators),
                            int(classifier_n_jobs),
                        )
                        family = "rival_imbalanced_ensemble"
                        oversampler_label = "internal"
                        classifier_label = classifier
                        sampling = "internal"

                    elapsed = round(float(time.perf_counter() - started), 6)
                    logical = logical_method_name(
                        kind,
                        classifier if kind == "rival" else oversampler,
                    )
                    screen_backbone = SCREEN_BACKBONES.get(context, context)
                    shared_shadow_runtime = (
                        float(mgvae_shadow[2]["runtime_seconds"])
                        if (
                            mgvae_shadow is not None
                            and oversampler_label == "mgvae"
                        )
                        else 0.0
                    )
                    if oversampler_label == "mgvae":
                        notes += (
                            f"; original Aiqz/MGVAE commit "
                            f"cad386bd2b3a90f6b740cbdf5f0cec8834102ea5; "
                            f"epochs={int(mgvae_epochs)}; one fold-level "
                            f"shadow shared between roles and backbones"
                        )
                    rows.append(
                        {
                            "dataset": record.name,
                            "dataset_source": record.source,
                            "seed": int(seed),
                            "fold": int(fold),
                            "logical_method": logical,
                            "method": method,
                            "paper_method_alias": method,
                            "method_family": family,
                            "classifier": classifier_label,
                            "screen_backbone": screen_backbone,
                            "backbone_context": context,
                            "oversampler": oversampler_label,
                            "sampling_strategy": sampling,
                            **compute_metrics(y_test, y_pred, y_score),
                            "n_train_original": int(len(y_train)),
                            "n_test": int(len(y_test)),
                            "n_train_resampled": int(n_resampled),
                            "n_generated": int(n_generated),
                            "runtime_seconds": elapsed,
                            "shared_shadow_runtime_seconds": round(
                                shared_shadow_runtime,
                                6,
                            ),
                            "mgvae_epochs": (
                                int(mgvae_epochs)
                                if oversampler_label == "mgvae"
                                else 0
                            ),
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
