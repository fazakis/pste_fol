from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from .oversampling import make_oversampler
from .utils import logit, positive_scores, sigmoid, to_numpy_xy


def _local_smoothed_priors(
    X_reference,
    y_reference,
    X_query,
    *,
    k: int,
    smooth: float,
    global_prior: float,
) -> np.ndarray:
    """Estimate a smoothed local minority prior using training data only."""

    X_reference = np.asarray(X_reference, dtype=float)
    y_reference = np.asarray(y_reference, dtype=int)
    X_query = np.asarray(X_query, dtype=float)
    n_reference = len(y_reference)
    if n_reference == 0:
        prior = float(np.clip(global_prior, 1e-6, 1.0 - 1e-6))
        return np.full(len(X_query), prior, dtype=float)

    k_effective = max(1, min(int(k), n_reference))
    neighbours = NearestNeighbors(n_neighbors=k_effective, algorithm="auto").fit(X_reference)
    indices = neighbours.kneighbors(X_query, return_distance=False)
    minority_counts = np.sum(y_reference[indices] == 1, axis=1).astype(float)
    smooth = max(0.0, float(smooth))
    global_prior = float(np.clip(global_prior, 1e-6, 1.0 - 1e-6))
    return np.clip(
        (minority_counts + smooth * global_prior) / (float(k_effective) + smooth),
        1e-6,
        1.0 - 1e-6,
    )


def _local_class_confidence(
    X_reference,
    y_reference,
    X_query,
    *,
    k: int,
    min_class_count: float,
) -> np.ndarray:
    """Return the local both-class evidence factor used by the PSTE gate."""

    X_reference = np.asarray(X_reference, dtype=float)
    y_reference = np.asarray(y_reference, dtype=int)
    X_query = np.asarray(X_query, dtype=float)
    n_reference = len(y_reference)
    if n_reference == 0:
        return np.zeros(len(X_query), dtype=float)

    k_effective = max(1, min(int(k), n_reference))
    neighbours = NearestNeighbors(n_neighbors=k_effective, algorithm="auto").fit(X_reference)
    indices = neighbours.kneighbors(X_query, return_distance=False)
    minority_counts = np.sum(y_reference[indices] == 1, axis=1).astype(float)
    majority_counts = float(k_effective) - minority_counts
    threshold = max(1e-6, float(min_class_count))
    return np.clip(np.minimum(minority_counts, majority_counts) / threshold, 0.0, 1.0)


def _mean_knn_distance(X_reference, X_query, *, k: int) -> np.ndarray:
    X_reference = np.asarray(X_reference, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    if X_reference.ndim != 2 or len(X_reference) == 0:
        return np.full(len(X_query), np.nan, dtype=float)
    k_effective = max(1, min(int(k), len(X_reference)))
    neighbours = NearestNeighbors(n_neighbors=k_effective, algorithm="auto").fit(X_reference)
    distances = neighbours.kneighbors(X_query, return_distance=True)[0]
    return np.mean(distances, axis=1)


def _support_reliability_scale(
    X_reference_queries,
    X_shadow_support,
    X_majority,
    *,
    k: int,
) -> float:
    """Estimate the geometric-gate scale from training data only."""

    X_reference_queries = np.asarray(X_reference_queries, dtype=float)
    d_shadow = _mean_knn_distance(
        X_shadow_support,
        X_reference_queries,
        k=k,
    )
    d_majority = _mean_knn_distance(
        X_majority,
        X_reference_queries,
        k=k,
    )
    contrast = d_majority - d_shadow
    finite_contrast = contrast[np.isfinite(contrast)]
    scale = float(np.nanmedian(np.abs(finite_contrast))) if len(finite_contrast) else 1.0
    if not np.isfinite(scale) or scale < 1e-9:
        pooled = np.concatenate(
            [d_shadow[np.isfinite(d_shadow)], d_majority[np.isfinite(d_majority)]]
        )
        scale = float(np.nanmedian(pooled)) if len(pooled) else 1.0
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return scale


def _support_reliability_gate(
    X_query,
    X_shadow_support,
    X_majority,
    *,
    scale: float,
    k: int,
) -> np.ndarray:
    """Measure query support using a scale fitted on training data."""

    X_query = np.asarray(X_query, dtype=float)
    d_shadow = _mean_knn_distance(X_shadow_support, X_query, k=k)
    d_majority = _mean_knn_distance(X_majority, X_query, k=k)
    if np.all(~np.isfinite(d_shadow)) or np.all(~np.isfinite(d_majority)):
        return np.ones(len(X_query), dtype=float)

    contrast = d_majority - d_shadow
    finite = np.isfinite(contrast)
    gate = np.ones(len(X_query), dtype=float)
    scale = float(scale)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    gate[finite] = sigmoid(2.0 * contrast[finite] / scale)
    return np.clip(gate, 0.0, 1.0)


class PSTEClassifier(ClassifierMixin, BaseEstimator):
    """Fixed confidence-gated Prior--Shadow Tree Ensemble (PSTE).

    The retained PSTE algorithm has no inner validation and no per-dataset
    selector. It divides one total tree budget between:

    * an original-prior branch trained on the untouched fold; and
    * a shadow branch trained on an oversampled copy of that fold.

    The default 200-tree configuration assigns 133 trees to the original branch
    and 67 to the shadow branch. Before probability-space blending, the shadow
    score receives the manuscript's local, clipped, damped, confidence-gated
    prior correction.

    Parameters
    ----------
    base_estimator:
        A scikit-learn-compatible tree classifier. If it exposes
        ``n_estimators``, ``max_iter``, or ``iterations``, that budget is split
        across the two branches.
    oversampler:
        An oversampler name or an object implementing ``fit_resample(X, y)``.
    total_estimators:
        Total branch budget. The retained experiment uses 200.
    shadow_fraction:
        Fraction assigned to the shadow branch. The retained value is 1/3.
    correction_strength:
        Damping factor gamma. The retained value is 0.15.
    local_prior_neighbors, local_prior_smoothing, local_shift_clip:
        Constants for the smoothed and clipped local prior contrast. Retained
        values are 31, 10, and 2.
    support_neighbors, min_local_class_count:
        Constants for the geometric and both-class confidence gates. Retained
        values are 7 and 3.
    blend_mode:
        ``"prob"`` is the retained and published method. ``"logit"`` is
        exposed only for explicitly custom studies and is not part of the
        locked benchmark or packaged reference results.
    """

    def __init__(
        self,
        base_estimator,
        oversampler="fast_outward_ladder",
        *,
        oversampler_kwargs: dict | None = None,
        sampling_strategy: float = 1.0,
        total_estimators: int | None = 200,
        shadow_fraction: float = 1.0 / 3.0,
        correction_strength: float = 0.15,
        local_prior_neighbors: int = 31,
        local_prior_smoothing: float = 10.0,
        local_shift_clip: float = 2.0,
        support_neighbors: int = 7,
        min_local_class_count: float = 3.0,
        blend_mode: str = "prob",
        random_state: int | None = 42,
        n_jobs: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.oversampler = oversampler
        self.oversampler_kwargs = oversampler_kwargs
        self.sampling_strategy = sampling_strategy
        self.total_estimators = total_estimators
        self.shadow_fraction = shadow_fraction
        self.correction_strength = correction_strength
        self.local_prior_neighbors = local_prior_neighbors
        self.local_prior_smoothing = local_prior_smoothing
        self.local_shift_clip = local_shift_clip
        self.support_neighbors = support_neighbors
        self.min_local_class_count = min_local_class_count
        self.blend_mode = blend_mode
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
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _allocate_budget(
        shadow_fraction: float,
        total_estimators: int | None,
    ) -> tuple[int | None, int | None]:
        if total_estimators is None:
            return None, None
        total = max(2, int(total_estimators))
        alpha = float(np.clip(shadow_fraction, 0.0, 1.0))
        shadow = int(round(total * alpha))
        shadow = min(max(1, shadow), total - 1)
        return total - shadow, shadow

    def _clone_estimator(self, seed: int, budget: int | None):
        estimator = clone(self.base_estimator)
        params = estimator.get_params()
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
            estimator.set_params(**updates)
        return estimator

    def _make_oversampler(self, seed: int):
        kwargs = dict(self.oversampler_kwargs or {})
        if isinstance(self.oversampler, str) or self.oversampler is None:
            return make_oversampler(
                self.oversampler,
                sampling_strategy=float(self.sampling_strategy),
                random_state=int(seed),
                **kwargs,
            )
        sampler = copy.deepcopy(self.oversampler)
        if not hasattr(sampler, "fit_resample"):
            raise TypeError("oversampler must be a name or implement fit_resample(X, y)")
        if hasattr(sampler, "random_state"):
            sampler.random_state = int(seed)
        if hasattr(sampler, "sampling_strategy"):
            sampler.sampling_strategy = float(self.sampling_strategy)
        for key, value in kwargs.items():
            if hasattr(sampler, key):
                setattr(sampler, key, value)
        return sampler

    def _fit_from_shadow_arrays(
        self,
        X,
        y,
        X_shadow,
        y_shadow,
        *,
        shadow_support=None,
        sampler=None,
        n_generated: int | None = None,
        sampler_warning: str = "",
    ):
        X, y = to_numpy_xy(X, y)
        X_shadow, y_shadow = to_numpy_xy(X_shadow, y_shadow)
        counts = np.bincount(y, minlength=2)
        shadow_counts = np.bincount(y_shadow, minlength=2)
        if np.any(counts == 0):
            raise ValueError("PSTE requires both binary classes in the training data")
        if np.any(shadow_counts == 0):
            raise ValueError("PSTE shadow data must contain both binary classes")
        if X_shadow.shape[1] != X.shape[1]:
            raise ValueError(
                "original and shadow data must have the same feature count"
            )
        if str(self.blend_mode) not in {"prob", "logit"}:
            raise ValueError("blend_mode must be 'prob' or 'logit'")

        seed = 0 if self.random_state is None else int(self.random_state)
        total_budget = self._total_budget_from_estimator()
        original_budget, shadow_budget = self._allocate_budget(
            self.shadow_fraction,
            total_budget,
        )
        self.original_estimator_ = self._clone_estimator(seed + 101, original_budget)
        self.original_estimator_.fit(X, y)
        self.shadow_estimator_ = self._clone_estimator(seed + 303, shadow_budget)
        self.shadow_estimator_.fit(X_shadow, y_shadow)

        if shadow_support is None:
            shadow_support = np.asarray(X_shadow[y_shadow == 1], dtype=float)
        else:
            shadow_support = np.asarray(shadow_support, dtype=float)
            if (
                shadow_support.ndim != 2
                or shadow_support.shape[1] != X.shape[1]
                or len(shadow_support) == 0
            ):
                raise ValueError(
                    "shadow_support must be a non-empty 2D array with the "
                    "same feature count as X"
                )

        self.sampler_ = sampler
        self.X_original_ = np.asarray(X, dtype=float)
        self.y_original_ = np.asarray(y, dtype=int)
        self.X_shadow_ = np.asarray(X_shadow, dtype=float)
        self.y_shadow_ = np.asarray(y_shadow, dtype=int)
        self.X_shadow_support_ = shadow_support
        self.X_original_majority_ = np.asarray(X[y == 0], dtype=float)
        self.support_scale_ = _support_reliability_scale(
            self.X_original_,
            self.X_shadow_support_,
            self.X_original_majority_,
            k=int(self.support_neighbors),
        )
        self.target_prior_ = float(np.mean(y == 1))
        self.shadow_source_prior_ = float(np.mean(y_shadow == 1))
        self.original_budget_ = original_budget
        self.shadow_budget_ = shadow_budget
        self.n_shadow_train_ = int(len(y_shadow))
        self.n_resampled_ = int(len(y_shadow))
        self.total_branch_training_rows_ = int(len(y) + len(y_shadow))
        if n_generated is None:
            n_generated = max(
                0,
                int(np.sum(y_shadow == 1) - np.sum(y == 1)),
            )
        self.n_generated_ = int(n_generated)
        self.sampler_warning_ = str(sampler_warning or "")
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = int(X.shape[1])

        # Compatibility/introspection attributes make the retained fixed
        # protocol explicit to callers of earlier adaptive implementations.
        self.selected_alpha_ = float(np.clip(self.shadow_fraction, 0.0, 1.0))
        self.selected_mode_ = str(self.blend_mode)
        self.selection_reason_ = "fixed_protocol_no_inner_validation"
        self.inner_cv_splits_ = 0
        self.validation_summary_ = {}
        self.candidate_summary_ = []
        return self

    @staticmethod
    def _shadow_support_from_sampler(sampler, X_shadow, y_shadow):
        parts = []
        for attribute in ("selected_base_points_", "selected_ladder_points_"):
            points = np.asarray(
                getattr(
                    sampler,
                    attribute,
                    np.empty((0, np.asarray(X_shadow).shape[1])),
                ),
                dtype=float,
            )
            if (
                points.ndim == 2
                and len(points)
                and points.shape[1] == np.asarray(X_shadow).shape[1]
            ):
                parts.append(points)
        if parts:
            return np.vstack(parts)
        return np.asarray(X_shadow, dtype=float)[
            np.asarray(y_shadow, dtype=int) == 1
        ]

    def fit(self, X, y):
        """Fit PSTE, including construction of the oversampled shadow fold."""

        X, y = to_numpy_xy(X, y)
        seed = 0 if self.random_state is None else int(self.random_state)
        sampler = self._make_oversampler(seed + 202)
        X_shadow, y_shadow = sampler.fit_resample(X, y)
        X_shadow, y_shadow = to_numpy_xy(X_shadow, y_shadow)
        return self._fit_from_shadow_arrays(
            X,
            y,
            X_shadow,
            y_shadow,
            shadow_support=self._shadow_support_from_sampler(
                sampler,
                X_shadow,
                y_shadow,
            ),
            sampler=sampler,
            n_generated=int(
                getattr(
                    sampler,
                    "n_generated_",
                    max(0, int(np.sum(y_shadow == 1) - np.sum(y == 1))),
                )
            ),
            sampler_warning=str(getattr(sampler, "warning_", "") or ""),
        )

    def fit_with_shadow(
        self,
        X,
        y,
        X_shadow,
        y_shadow,
        *,
        shadow_support=None,
        n_generated: int | None = None,
        sampler_warning: str = "",
    ):
        """Fit from a precomputed training-fold shadow dataset.

        The locked MGVAE protocol uses this path so one expensive fold-level
        shadow dataset can be shared between standalone and PSTE roles and
        across the three downstream backbones. The outer test fold is never
        used to construct the supplied shadow data.
        """

        return self._fit_from_shadow_arrays(
            X,
            y,
            X_shadow,
            y_shadow,
            shadow_support=shadow_support,
            sampler=None,
            n_generated=n_generated,
            sampler_warning=sampler_warning,
        )

    def _correct_shadow_scores(self, X, raw_shadow_score):
        pi_original = _local_smoothed_priors(
            self.X_original_,
            self.y_original_,
            X,
            k=int(self.local_prior_neighbors),
            smooth=float(self.local_prior_smoothing),
            global_prior=self.target_prior_,
        )
        pi_shadow = _local_smoothed_priors(
            self.X_shadow_,
            self.y_shadow_,
            X,
            k=int(self.local_prior_neighbors),
            smooth=float(self.local_prior_smoothing),
            global_prior=self.shadow_source_prior_,
        )
        shift_clip = max(0.0, float(self.local_shift_clip))
        local_shift = np.clip(logit(pi_original) - logit(pi_shadow), -shift_clip, shift_clip)

        support_gate = _support_reliability_gate(
            X,
            self.X_shadow_support_,
            self.X_original_majority_,
            scale=self.support_scale_,
            k=int(self.support_neighbors),
        )
        original_confidence = _local_class_confidence(
            self.X_original_,
            self.y_original_,
            X,
            k=int(self.local_prior_neighbors),
            min_class_count=float(self.min_local_class_count),
        )
        shadow_confidence = _local_class_confidence(
            self.X_shadow_,
            self.y_shadow_,
            X,
            k=int(self.local_prior_neighbors),
            min_class_count=float(self.min_local_class_count),
        )
        class_confidence = np.minimum(original_confidence, shadow_confidence)
        gate = support_gate * class_confidence
        corrected = sigmoid(
            logit(raw_shadow_score)
            + float(self.correction_strength) * gate * local_shift
        )
        return corrected, local_shift, support_gate, class_confidence, gate

    def predict_components(self, X) -> dict[str, np.ndarray]:
        """Return branch scores and all correction terms for inspection."""

        check_is_fitted(self, ("original_estimator_", "shadow_estimator_"))
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_features_in_}); got {X.shape}"
            )
        original = positive_scores(self.original_estimator_, X)
        shadow_raw = positive_scores(self.shadow_estimator_, X)
        shadow_corrected, local_shift, support_gate, class_confidence, gate = (
            self._correct_shadow_scores(X, shadow_raw)
        )
        alpha = float(np.clip(self.shadow_fraction, 0.0, 1.0))
        if abs(alpha - (1.0 / 3.0)) <= 1e-15:
            if self.blend_mode == "logit":
                final = sigmoid(
                    (2.0 * logit(original) + logit(shadow_corrected)) / 3.0
                )
            else:
                final = (2.0 * original + shadow_corrected) / 3.0
        elif self.blend_mode == "logit":
            final = sigmoid((1.0 - alpha) * logit(original) + alpha * logit(shadow_corrected))
        else:
            final = (1.0 - alpha) * original + alpha * shadow_corrected
        return {
            "original": np.asarray(original, dtype=float),
            "shadow_raw": np.asarray(shadow_raw, dtype=float),
            "shadow_corrected": np.asarray(shadow_corrected, dtype=float),
            "local_shift": np.asarray(local_shift, dtype=float),
            "support_gate": np.asarray(support_gate, dtype=float),
            "class_confidence": np.asarray(class_confidence, dtype=float),
            "gate": np.asarray(gate, dtype=float),
            "final": np.clip(np.asarray(final, dtype=float), 0.0, 1.0),
        }

    def predict_proba(self, X):
        score = self.predict_components(X)["final"]
        return np.column_stack([1.0 - score, score])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
