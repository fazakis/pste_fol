from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score
from sklearn.model_selection import StratifiedKFold

from .oversampling import make_oversampler
from .utils import (
    alpha_to_ratio,
    blend_scores,
    clip01,
    positive_scores,
    precision_at_top_k,
    prior_correct_binary_probability,
    to_numpy_xy,
)

PSTE_DEFAULT_ALPHAS = [round(i / 50.0, 2) for i in range(0, 51)]
PSTE_DEFAULT_MODES = ["prob", "logit", "priorcorr", "priorcorr_logit"]
PSTE_BLEND_MODES = {"prob", "logit", "priorcorr", "priorcorr_logit", "rank", "priorcorr_rank"}


def _parse_alpha_grid(values: Iterable[float] | str | None) -> list[float]:
    if values is None:
        raw = PSTE_DEFAULT_ALPHAS
    elif isinstance(values, str):
        raw = [float(v) for v in values.replace(",", " ").split() if v.strip()]
    else:
        raw = list(values)
    out = sorted({round(float(np.clip(float(a), 0.0, 1.0)), 6) for a in raw})
    if not out:
        raise ValueError("PSTE alpha grid is empty")
    return out


def _parse_modes(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        raw = PSTE_DEFAULT_MODES
    elif isinstance(values, str):
        raw = [v for v in values.replace(",", " ").split() if v.strip()]
    else:
        raw = list(values)
    out: list[str] = []
    for mode in raw:
        mode = str(mode).strip()
        if mode not in PSTE_BLEND_MODES:
            raise ValueError(f"unsupported PSTE blend mode {mode!r}; choose from {sorted(PSTE_BLEND_MODES)}")
        if mode not in out:
            out.append(mode)
    if not out:
        raise ValueError("PSTE mode grid is empty")
    return out


def _constraint_enabled(value) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) >= 0.0


@dataclass
class _BranchScores:
    original: np.ndarray
    shadow: np.ndarray
    source_prior: float
    target_prior: float
    n_resampled: int
    n_generated: int
    warning: str


class PSTEClassifier(BaseEstimator, ClassifierMixin):
    """Prior-stable tree ensemble wrapper (PSTE).

    PSTE is a two-branch wrapper for tree classifiers:

    * the original-prior branch fits a clone of the supplied tree classifier on
      the original training fold;
    * the shadow branch fits a clone on an oversampled version of the same
      training fold; and
    * nested inner validation chooses a leakage-free blend of the two branch
      score streams, constrained by prior drift, Brier degradation, and
      precision@top-k loss.

    Parameters
    ----------
    base_estimator:
        Any scikit-learn-compatible tree classifier with ``fit`` and either
        ``predict_proba`` or ``decision_function``. Ensemble sizes are split
        automatically when the estimator exposes ``n_estimators``, ``max_iter``,
        or ``iterations``.
    oversampler:
        Oversampler name (e.g. ``"smote"`` or ``"fast_outward_ladder"``) or an
        object implementing ``fit_resample(X, y)``.
    oversampler_kwargs:
        Optional keyword arguments passed to named oversampler constructors, or
        copied onto oversampler objects when matching attributes exist.
    alpha_grid:
        Candidate shadow-score fractions. ``alpha=1/3`` equals an approximate
        original:shadow score ratio of 2:1.
    modes:
        Candidate blend/correction modes: ``prob``, ``logit``, ``priorcorr``,
        ``priorcorr_logit``. ``rank`` and ``priorcorr_rank`` are available for
        experimentation.
    """

    def __init__(
        self,
        base_estimator,
        oversampler="fast_outward_ladder",
        *,
        oversampler_kwargs: dict | None = None,
        sampling_strategy: float = 1.0,
        total_estimators: int | None = None,
        alpha_grid: Iterable[float] | str | None = None,
        modes: Iterable[str] | str | None = None,
        default_alpha: float = 1.0 / 3.0,
        default_mode: str = "prob",
        prior_drift_max: float = 0.05,
        brier_degradation_max: float = 0.01,
        precision_topk_loss_max: float = 0.05,
        inner_cv_folds: int = 3,
        inner_cv_repeats: int = 2,
        confidence_z: float = 1.96,
        inner_budget_cap: int = 80,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.oversampler = oversampler
        self.oversampler_kwargs = oversampler_kwargs
        self.sampling_strategy = sampling_strategy
        self.total_estimators = total_estimators
        self.alpha_grid = alpha_grid
        self.modes = modes
        self.default_alpha = default_alpha
        self.default_mode = default_mode
        self.prior_drift_max = prior_drift_max
        self.brier_degradation_max = brier_degradation_max
        self.precision_topk_loss_max = precision_topk_loss_max
        self.inner_cv_folds = inner_cv_folds
        self.inner_cv_repeats = inner_cv_repeats
        self.confidence_z = confidence_z
        self.inner_budget_cap = inner_budget_cap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _total_budget_from_estimator(self) -> int | None:
        if self.total_estimators is not None:
            return int(self.total_estimators)
        try:
            params = self.base_estimator.get_params()
        except Exception:
            return None
        for key in ("n_estimators", "max_iter", "iterations"):
            if key in params and params[key] is not None:
                try:
                    return int(params[key])
                except Exception:
                    pass
        return None

    @staticmethod
    def _allocate_budget(alpha: float, total: int | None) -> tuple[int | None, int | None]:
        if total is None:
            return None, None
        total = max(2, int(total))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        shadow = int(round(total * alpha))
        shadow = min(max(1, shadow), total - 1)
        original = total - shadow
        return int(original), int(shadow)

    def _clone_estimator(self, seed: int, budget: int | None = None):
        est = clone(self.base_estimator)
        params = est.get_params()
        updates = {}
        if "random_state" in params:
            updates["random_state"] = int(seed)
        if self.n_jobs is not None and "n_jobs" in params:
            updates["n_jobs"] = int(self.n_jobs)
        if budget is not None:
            for key in ("n_estimators", "max_iter", "iterations"):
                if key in params:
                    updates[key] = max(1, int(budget))
                    break
        if updates:
            est.set_params(**updates)
        return est

    def _make_oversampler(self, seed: int):
        kwargs = dict(self.oversampler_kwargs or {})
        if isinstance(self.oversampler, str) or self.oversampler is None:
            return make_oversampler(self.oversampler, sampling_strategy=float(self.sampling_strategy), random_state=int(seed), **kwargs)
        sampler = copy.deepcopy(self.oversampler)
        if hasattr(sampler, "random_state"):
            sampler.random_state = int(seed)
        if hasattr(sampler, "sampling_strategy"):
            sampler.sampling_strategy = float(self.sampling_strategy)
        for key, value in kwargs.items():
            if hasattr(sampler, key):
                setattr(sampler, key, value)
        if not hasattr(sampler, "fit_resample"):
            raise TypeError("oversampler must be a name or implement fit_resample(X, y)")
        return sampler

    def _fit_shared_scores(self, X_train, y_train, X_score, seed: int, original_budget: int | None, shadow_budget: int | None) -> _BranchScores:
        original_est = self._clone_estimator(seed + 101, original_budget)
        original_est.fit(X_train, y_train)
        score_original = positive_scores(original_est, X_score)

        sampler = self._make_oversampler(seed + 202)
        X_shadow, y_shadow = sampler.fit_resample(X_train, y_train)
        shadow_est = self._clone_estimator(seed + 303, shadow_budget)
        shadow_est.fit(X_shadow, y_shadow)
        score_shadow = positive_scores(shadow_est, X_score)

        n_generated = getattr(sampler, "n_generated_", max(0, int(np.sum(y_shadow == 1) - np.sum(y_train == 1))))
        return _BranchScores(
            original=score_original,
            shadow=score_shadow,
            source_prior=float(np.mean(np.asarray(y_shadow, dtype=int) == 1)),
            target_prior=float(np.mean(np.asarray(y_train, dtype=int) == 1)),
            n_resampled=int(len(y_shadow)),
            n_generated=int(n_generated),
            warning=str(getattr(sampler, "warning_", "") or ""),
        )

    def _score_mode_alpha(self, scores: _BranchScores, mode: str, alpha: float):
        shadow = scores.shadow
        if mode in {"priorcorr", "priorcorr_logit", "priorcorr_rank"}:
            shadow = prior_correct_binary_probability(shadow, source_prior=scores.source_prior, target_prior=scores.target_prior)
        return blend_scores(scores.original, shadow, alpha, mode=mode)

    @staticmethod
    def _validation_stats(y_true, y_score) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=int)
        y_score = clip01(y_score)
        y_pred = (y_score >= 0.5).astype(int)
        return {
            "ap": float(average_precision_score(y_true, y_score)),
            "pred_prior": float(np.mean(y_score)),
            "true_prior": float(np.mean(y_true == 1)),
            "prior_drift": float(abs(np.mean(y_score) - np.mean(y_true == 1))),
            "brier": float(brier_score_loss(y_true, y_score)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "precision_topk": precision_at_top_k(y_true, y_score, int(np.sum(y_true == 1))),
        }

    @staticmethod
    def _mean(rows: list[dict[str, float]], key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    @staticmethod
    def _se(rows: list[dict[str, float]], key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        if len(vals) < 2:
            return float("inf")
        return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

    def fit(self, X, y):
        X, y = to_numpy_xy(X, y)
        self.classes_ = np.array([0, 1], dtype=int)
        self.alpha_grid_ = _parse_alpha_grid(self.alpha_grid)
        self.default_alpha_ = round(float(np.clip(self.default_alpha, 0.0, 1.0)), 6)
        if self.default_alpha_ not in self.alpha_grid_:
            self.alpha_grid_ = sorted({self.default_alpha_, *self.alpha_grid_})
        self.modes_ = _parse_modes(self.modes)
        self.default_mode_ = str(self.default_mode)
        if self.default_mode_ not in PSTE_BLEND_MODES:
            raise ValueError(f"unsupported default_mode={self.default_mode_!r}")
        if self.default_mode_ not in self.modes_:
            self.modes_ = [self.default_mode_] + self.modes_
        self.candidates_ = [(m, a) for m in self.modes_ for a in self.alpha_grid_]
        default_key = (self.default_mode_, self.default_alpha_)
        if default_key not in self.candidates_:
            self.candidates_ = [default_key] + self.candidates_

        total_budget = self._total_budget_from_estimator()
        counts = np.bincount(y.astype(int), minlength=2)
        min_count = int(counts.min())
        rows_by_key: dict[tuple[str, float], list[dict[str, float]]] = {key: [] for key in self.candidates_}
        base_rows: list[dict[str, float]] = []
        warnings_seen: list[str] = []
        split_count = 0

        if min_count >= 2:
            n_splits = min(max(2, int(self.inner_cv_folds)), min_count)
            n_repeats = max(1, int(self.inner_cv_repeats))
            inner_total = total_budget
            if inner_total is not None:
                inner_total = max(2, min(int(inner_total), int(self.inner_budget_cap)))
            max_original = max((self._allocate_budget(a, inner_total)[0] or 0) for a in self.alpha_grid_) if inner_total is not None else None
            max_shadow = max((self._allocate_budget(a, inner_total)[1] or 0) for a in self.alpha_grid_) if inner_total is not None else None
            for rep in range(n_repeats):
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int((self.random_state or 0) + 3404 + 7919 * rep))
                for fold, (sub_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
                    split_count += 1
                    X_sub, X_val = X[sub_idx], X[val_idx]
                    y_sub, y_val = y[sub_idx], y[val_idx]
                    scores = self._fit_shared_scores(X_sub, y_sub, X_val, int((self.random_state or 0) + 3600 + 1009 * rep + 37 * fold), max_original, max_shadow)
                    if scores.warning:
                        warnings_seen.append(scores.warning)
                    base_stats = self._validation_stats(y_val, scores.original)
                    base_rows.append(base_stats)
                    default_score = self._score_mode_alpha(scores, self.default_mode_, self.default_alpha_)
                    default_stats = self._validation_stats(y_val, default_score)
                    for mode, alpha in self.candidates_:
                        if (mode, alpha) == default_key:
                            stats = dict(default_stats)
                        else:
                            stats = self._validation_stats(y_val, self._score_mode_alpha(scores, mode, alpha))
                        stats["brier_degradation"] = float(stats["brier"] - base_stats["brier"])
                        stats["precision_loss"] = float(base_stats["precision"] - stats["precision"])
                        stats["precision_topk_loss"] = float(base_stats["precision_topk"] - stats["precision_topk"])
                        stats["ap_delta_vs_default"] = float(stats["ap"] - default_stats["ap"])
                        rows_by_key[(mode, alpha)].append(stats)
        else:
            n_splits = 0
            n_repeats = 0

        evaluated = []
        z = max(0.0, float(self.confidence_z))
        for key, rows in rows_by_key.items():
            if not rows:
                continue
            mode, alpha = key
            stats = {metric: self._mean(rows, metric) for metric in [
                "ap", "prior_drift", "brier", "precision", "precision_topk",
                "brier_degradation", "precision_loss", "precision_topk_loss", "ap_delta_vs_default",
            ]}
            stats["ap_lcb"] = stats["ap"] - z * self._se(rows, "ap")
            stats["ap_delta_lcb"] = stats["ap_delta_vs_default"] - z * self._se(rows, "ap_delta_vs_default")
            stats["prior_drift_ucb"] = stats["prior_drift"] + z * self._se(rows, "prior_drift")
            stats["brier_degradation_ucb"] = stats["brier_degradation"] + z * self._se(rows, "brier_degradation")
            stats["precision_topk_loss_ucb"] = stats["precision_topk_loss"] + z * self._se(rows, "precision_topk_loss")
            violations = 0
            if _constraint_enabled(self.prior_drift_max) and stats["prior_drift_ucb"] > float(self.prior_drift_max):
                violations += 1
            if _constraint_enabled(self.brier_degradation_max) and stats["brier_degradation_ucb"] > float(self.brier_degradation_max):
                violations += 1
            if _constraint_enabled(self.precision_topk_loss_max) and stats["precision_topk_loss_ucb"] > float(self.precision_topk_loss_max):
                violations += 1
            evaluated.append({"key": key, "mode": mode, "alpha": alpha, "stats": stats, "feasible": violations == 0, "violations": violations, "cv_splits": len(rows)})

        default_row = next((r for r in evaluated if r["key"] == default_key), None)
        feasible = [r for r in evaluated if r["feasible"]]
        positive_delta = [r for r in feasible if r["key"] != default_key and np.isfinite(r["stats"].get("ap_delta_lcb", np.nan)) and r["stats"]["ap_delta_lcb"] > 0.0]
        if default_row is not None and default_row["feasible"] and not positive_delta:
            selected = default_row
            reason = "default_feasible_no_positive_lcb_delta"
        elif positive_delta:
            selected = max(positive_delta, key=lambda r: (r["stats"].get("ap_delta_lcb", -np.inf), r["stats"].get("ap_lcb", -np.inf), -abs(r["alpha"] - self.default_alpha_)))
            reason = "adapted_positive_lcb_delta"
        elif feasible:
            selected = max(feasible, key=lambda r: (r["stats"].get("ap_lcb", -np.inf), r["stats"].get("ap", -np.inf), -abs(r["alpha"] - self.default_alpha_)))
            reason = "best_feasible_lcb_ap"
        else:
            selected = default_row or {"mode": self.default_mode_, "alpha": self.default_alpha_, "stats": {}, "feasible": False, "violations": -1, "cv_splits": 0}
            reason = "default_relaxed_or_too_few_samples"

        self.selected_mode_ = str(selected["mode"])
        self.selected_alpha_ = float(selected["alpha"])
        self.selected_ratio_ = alpha_to_ratio(self.selected_alpha_)
        self.selection_reason_ = reason
        self.inner_cv_splits_ = int(split_count)
        self.validation_summary_ = selected.get("stats", {})
        self.candidate_summary_ = evaluated

        original_budget, shadow_budget = self._allocate_budget(self.selected_alpha_, total_budget)
        self.original_estimator_ = self._clone_estimator(int((self.random_state or 0) + 101), original_budget)
        self.original_estimator_.fit(X, y)
        sampler = self._make_oversampler(int((self.random_state or 0) + 202))
        X_shadow, y_shadow = sampler.fit_resample(X, y)
        self.shadow_estimator_ = self._clone_estimator(int((self.random_state or 0) + 303), shadow_budget)
        self.shadow_estimator_.fit(X_shadow, y_shadow)
        self.shadow_source_prior_ = float(np.mean(np.asarray(y_shadow, dtype=int) == 1))
        self.target_prior_ = float(np.mean(np.asarray(y, dtype=int) == 1))
        self.n_resampled_ = int(len(y_shadow))
        self.n_generated_ = int(getattr(sampler, "n_generated_", max(0, int(np.sum(y_shadow == 1) - np.sum(y == 1)))))
        self.sampler_warning_ = str(getattr(sampler, "warning_", "") or "")
        if warnings_seen and not self.sampler_warning_:
            self.sampler_warning_ = "; ".join(sorted(set(warnings_seen))[:3])
        self.original_budget_ = original_budget
        self.shadow_budget_ = shadow_budget
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        score_original = positive_scores(self.original_estimator_, X)
        score_shadow = positive_scores(self.shadow_estimator_, X)
        if self.selected_mode_ in {"priorcorr", "priorcorr_logit", "priorcorr_rank"}:
            score_shadow = prior_correct_binary_probability(score_shadow, self.shadow_source_prior_, self.target_prior_)
        p = clip01(blend_scores(score_original, score_shadow, self.selected_alpha_, self.selected_mode_))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
