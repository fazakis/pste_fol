from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .utils import EPS, normalize01, synthetic_needed, to_numpy_xy


def _candidate_key(x: np.ndarray, decimals: int = 10) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(x, dtype=float), decimals=decimals).tolist())


def _unit(v: np.ndarray) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < EPS:
        return None
    return v / n


def _bounds_for(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    margin = 0.01 * span
    return lo - margin, hi + margin


def _within_bounds(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    return bool(np.all(x >= lo) and np.all(x <= hi))


def _simple_smote(X_min: np.ndarray, n: int, rng: np.random.RandomState, k_neighbors: int = 5, candidate_multiplier: int = 1) -> np.ndarray:
    X_min = np.asarray(X_min, dtype=float)
    n = int(n)
    if n <= 0 or len(X_min) == 0:
        return np.empty((0, X_min.shape[1] if X_min.ndim == 2 else 0), dtype=float)
    if len(X_min) == 1:
        jitter = rng.normal(0.0, 1e-6, size=(n, X_min.shape[1]))
        return X_min[0].reshape(1, -1) + jitter
    k = min(len(X_min), max(2, int(k_neighbors) + 1))
    nn = NearestNeighbors(n_neighbors=k).fit(X_min)
    out = []
    for _ in range(max(n * int(candidate_multiplier), n)):
        i = int(rng.randint(len(X_min)))
        neigh = nn.kneighbors(X_min[i].reshape(1, -1), return_distance=False)[0]
        choices = [int(j) for j in neigh if int(j) != i]
        if not choices:
            continue
        j = int(choices[int(rng.randint(len(choices)))])
        lam = float(rng.uniform(0.05, 0.95))
        out.append(X_min[i] + lam * (X_min[j] - X_min[i]))
        if len(out) >= n:
            break
    return np.vstack(out) if out else np.empty((0, X_min.shape[1]))


class NoneOversampler:
    """No-op object with the same interface as oversamplers."""

    def __init__(self, sampling_strategy: float = 1.0, random_state: int | None = None, **kwargs):
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.n_generated_ = 0
        self.warning_ = ""

    def fit_resample(self, X, y):
        X, y = to_numpy_xy(X, y)
        self.n_generated_ = 0
        self.warning_ = ""
        return X.copy(), y.copy()


class SafeImblearnOversampler:
    """Safe wrapper for standard imbalanced-learn oversamplers.

    If a fold is too small or a method raises, the original training set is
    returned and the warning is stored. This mirrors the manuscript benchmark's
    conservative handling of tiny-minority folds.
    """

    def __init__(self, method: str, sampling_strategy: float = 1.0, random_state: int | None = None, **kwargs):
        self.method = str(method).lower()
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.kwargs = dict(kwargs)
        self.n_generated_ = 0
        self.warning_ = ""

    def _build_sampler(self, n_min: int):
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE

        common = dict(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        k = max(1, min(5, int(n_min) - 1))
        if self.method in {"random_over_sampler", "ros", "random"}:
            return RandomOverSampler(**common)
        if self.method == "smote":
            return SMOTE(k_neighbors=k, **common)
        if self.method in {"smote_tomek", "smotetomek"}:
            return SMOTETomek(smote=SMOTE(k_neighbors=k, **common), **common)
        if self.method in {"borderline_smote", "borderline"}:
            return BorderlineSMOTE(k_neighbors=k, m_neighbors=max(1, min(10, int(n_min) - 1)), **common)
        if self.method == "adasyn":
            return ADASYN(n_neighbors=k, **common)
        if self.method in {"kmeans_smote", "kmeans"}:
            return KMeansSMOTE(k_neighbors=k, cluster_balance_threshold=0.01, **common)
        raise ValueError(f"unknown imbalanced-learn oversampler: {self.method}")

    def fit_resample(self, X, y):
        X, y = to_numpy_xy(X, y)
        self.n_generated_ = 0
        self.warning_ = ""
        need = synthetic_needed(y, self.sampling_strategy)
        n_min = int(np.sum(y == 1))
        n_maj = int(np.sum(y == 0))
        if need <= 0:
            return X.copy(), y.copy()
        if n_min < 2 or n_maj < 1:
            self.warning_ = "too few samples for oversampling"
            return X.copy(), y.copy()
        try:
            sampler = self._build_sampler(n_min)
            X_res, y_res = sampler.fit_resample(X, y)
            X_res = np.asarray(X_res, dtype=float)
            y_res = np.asarray(y_res, dtype=int)
            self.n_generated_ = max(0, int(np.sum(y_res == 1) - np.sum(y == 1)))
            return X_res, y_res
        except Exception as exc:  # pragma: no cover - data-dependent safeguard
            self.warning_ = f"{self.method} failed: {type(exc).__name__}: {exc}"
            warnings.warn(self.warning_, RuntimeWarning)
            return X.copy(), y.copy()


class DeepSMOTEOversampler:
    """Standalone tabular DeepSMOTE-style latent-space oversampler.

    This implementation intentionally avoids any external paper-code classes. It
    learns a compact PCA latent representation on the training fold, performs
    SMOTE in that latent space, and decodes by inverse PCA transformation. It is
    provided so the reproducibility repo contains a runnable DeepSMOTE-family
    rival without depending on a separate repository.
    """

    def __init__(self, sampling_strategy: float = 1.0, random_state: int | None = None, latent_dim: int = 10, **kwargs):
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.latent_dim = int(latent_dim)
        self.n_generated_ = 0
        self.warning_ = ""

    def fit_resample(self, X, y):
        X, y = to_numpy_xy(X, y)
        self.n_generated_ = 0
        self.warning_ = ""
        need = synthetic_needed(y, self.sampling_strategy)
        if need <= 0:
            return X.copy(), y.copy()
        X = np.nan_to_num(X.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        X_min = X[y == 1]
        if len(X_min) < 2:
            self.warning_ = "too few minority samples for DeepSMOTE-style latent sampling"
            return X.copy(), y.copy()
        rng = np.random.RandomState(self.random_state)
        try:
            n_components = max(1, min(self.latent_dim, X.shape[1], len(X) - 1))
            pca = PCA(n_components=n_components, random_state=self.random_state)
            Z = pca.fit_transform(X)
            Z_min = Z[y == 1]
            Z_syn = _simple_smote(Z_min, need, rng)
            X_syn = pca.inverse_transform(Z_syn)
            lo, hi = _bounds_for(X)
            X_syn = np.clip(X_syn, lo, hi)
        except Exception as exc:
            self.warning_ = f"latent PCA fallback used after {type(exc).__name__}: {exc}"
            X_syn = _simple_smote(X_min, need, rng)
        self.n_generated_ = int(len(X_syn))
        return np.vstack([X, X_syn]), np.concatenate([y, np.ones(self.n_generated_, dtype=y.dtype)])


class MGVAEOversampler:
    """Standalone majority-guided VAE-inspired rival oversampler.

    The paper benchmark used MGVAE as an external rival. This self-contained
    implementation captures the same high-level idea for reproducible runs:
    minority latent interpolation followed by a small displacement away from the
    nearest majority samples. No external MGVAE classes are required.
    """

    def __init__(self, sampling_strategy: float = 1.0, random_state: int | None = None, majority_push: float = 0.15, **kwargs):
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.majority_push = float(majority_push)
        self.n_generated_ = 0
        self.warning_ = ""

    def fit_resample(self, X, y):
        X, y = to_numpy_xy(X, y)
        self.n_generated_ = 0
        self.warning_ = ""
        need = synthetic_needed(y, self.sampling_strategy)
        if need <= 0:
            return X.copy(), y.copy()
        X = np.nan_to_num(X.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        X_min = X[y == 1]
        X_maj = X[y == 0]
        if len(X_min) < 2 or len(X_maj) < 1:
            self.warning_ = "too few samples for MGVAE-style sampling"
            return X.copy(), y.copy()
        rng = np.random.RandomState(self.random_state)
        syn = _simple_smote(X_min, need, rng)
        maj_nn = NearestNeighbors(n_neighbors=1).fit(X_maj)
        _, idx = maj_nn.kneighbors(syn, return_distance=True)
        nearest_maj = X_maj[idx[:, 0]]
        away = syn - nearest_maj
        norm = np.linalg.norm(away, axis=1, keepdims=True)
        local_scale = np.median(np.linalg.norm(X_min - np.mean(X_min, axis=0), axis=1))
        local_scale = float(local_scale) if np.isfinite(local_scale) and local_scale > EPS else 1.0
        syn = syn + self.majority_push * local_scale * away / np.maximum(norm, EPS)
        lo, hi = _bounds_for(X)
        syn = np.clip(syn, lo, hi)
        self.n_generated_ = int(len(syn))
        return np.vstack([X, syn]), np.concatenate([y, np.ones(self.n_generated_, dtype=y.dtype)])


@dataclass
class _LadderCandidate:
    x: np.ndarray
    anchor: np.ndarray
    frac: float
    depth: float
    d_min: float
    d_maj: float
    ratio: float


class FastOutwardLadderOversampler:
    """Standalone Fast Outward Ladder oversampler.

    Fast Outward Ladder creates minority synthetic samples in two parts:

    1. a conventional SMOTE seed cloud inside minority support; and
    2. outward ladder rungs that start near the minority/boundary side and move
       away from nearby majority mass, while retaining distance, tube, boundary,
       and duplicate checks.

    This file is deliberately standalone. It does not inherit from, import, or
    require any of the exploratory ancestor classes used during method design.
    """

    def __init__(
        self,
        sampling_strategy: float = 1.0,
        random_state: int | None = None,
        *,
        base_smote_fraction: float = 0.40,
        ladder_step_fracs: Iterable[float] = (0.12, 0.24, 0.38, 0.54),
        max_ladder_depth_factor: float = 1.05,
        candidate_multiplier: int = 6,
        max_candidates: int = 8000,
        minority_neighbors: int = 6,
        majority_neighbors: int = 6,
        inward_target_mode: str = "nearest_majority_centroid",
        boundary_preference: float = 1.35,
        min_boundary_ratio: float = 0.75,
        max_boundary_ratio: float = 5.0,
        safety_factor: float = 2.40,
        tube_radius_factor: float = 0.85,
        boundary_weight: float = 0.30,
        plausibility_weight: float = 0.28,
        coverage_weight: float = 0.45,
        intrusion_weight: float = 0.42,
        redundancy_weight: float = 0.06,
        allow_minor_relaxation: bool = True,
        **kwargs,
    ):
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.base_smote_fraction = float(base_smote_fraction)
        self.ladder_step_fracs = tuple(float(f) for f in ladder_step_fracs)
        self.max_ladder_depth_factor = float(max_ladder_depth_factor)
        self.candidate_multiplier = int(candidate_multiplier)
        self.max_candidates = int(max_candidates)
        self.minority_neighbors = int(minority_neighbors)
        self.majority_neighbors = int(majority_neighbors)
        self.inward_target_mode = str(inward_target_mode)
        self.boundary_preference = float(boundary_preference)
        self.min_boundary_ratio = float(min_boundary_ratio)
        self.max_boundary_ratio = float(max_boundary_ratio)
        self.safety_factor = float(safety_factor)
        self.tube_radius_factor = float(tube_radius_factor)
        self.boundary_weight = float(boundary_weight)
        self.plausibility_weight = float(plausibility_weight)
        self.coverage_weight = float(coverage_weight)
        self.intrusion_weight = float(intrusion_weight)
        self.redundancy_weight = float(redundancy_weight)
        self.allow_minor_relaxation = bool(allow_minor_relaxation)
        self.n_generated_ = 0
        self.n_base_smote_ = 0
        self.n_selected_ladder_ = 0
        self.warning_ = ""

    def _basis(self, anchor: np.ndarray, X_min: np.ndarray, min_nn: NearestNeighbors) -> tuple[np.ndarray, np.ndarray, float]:
        k = min(len(X_min), max(3, int(self.minority_neighbors)))
        _, inds = min_nn.kneighbors(anchor.reshape(1, -1), n_neighbors=k, return_distance=True)
        neigh = X_min[inds[0]]
        center = neigh.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(neigh - center, full_matrices=False)
            tangent = vt[0]
        except Exception:
            tangent = np.ones(anchor.size)
        tangent = _unit(tangent)
        if tangent is None:
            tangent = np.ones(anchor.size) / np.sqrt(anchor.size)
        scale = float(np.mean(np.linalg.norm(neigh - anchor, axis=1)))
        if not np.isfinite(scale) or scale < EPS:
            scale = 1.0
        return center, tangent, scale

    def _outward_direction(self, x: np.ndarray, X_maj: np.ndarray, maj_nn: NearestNeighbors) -> np.ndarray | None:
        mode = str(self.inward_target_mode).lower()
        if mode == "all_majority_centroid":
            target = X_maj.mean(axis=0)
        elif mode == "nearest_majority":
            target = X_maj[maj_nn.kneighbors(x.reshape(1, -1), n_neighbors=1, return_distance=False)[0, 0]]
        else:
            k = min(len(X_maj), max(1, int(self.majority_neighbors)))
            d, ind = maj_nn.kneighbors(x.reshape(1, -1), n_neighbors=k, return_distance=True)
            w = 1.0 / (d[0] + EPS)
            target = np.average(X_maj[ind[0]], axis=0, weights=w)
        inward = _unit(target - x)
        if inward is None:
            return None
        return -inward

    def _ladder_candidates(self, X, X_min, X_maj, base, n_target, rng) -> list[_LadderCandidate]:
        if n_target <= 0:
            return []
        lo, hi = _bounds_for(X)
        anchors = np.vstack([X_min, base]) if len(base) else X_min.copy()
        min_nn = NearestNeighbors(n_neighbors=min(len(X_min), max(2, self.minority_neighbors))).fit(X_min)
        maj_nn = NearestNeighbors(n_neighbors=min(len(X_maj), max(1, self.majority_neighbors))).fit(X_maj)
        real_nn = NearestNeighbors(n_neighbors=1).fit(X)

        anchor_infos = []
        for i, a in enumerate(anchors):
            if not np.all(np.isfinite(a)):
                continue
            dmin = float(min_nn.kneighbors(a.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0, 0])
            dmaj = float(maj_nn.kneighbors(a.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0, 0])
            ratio = dmaj / (dmin + EPS)
            if not (dmaj > dmin and ratio <= self.max_boundary_ratio):
                continue
            _center, tangent, local_scale = self._basis(a, X_min, min_nn)
            anchor_infos.append((abs(ratio - self.boundary_preference), -local_scale, i, a, tangent, local_scale))
        anchor_infos.sort()

        max_raw = min(int(self.max_candidates), max(int(n_target) * int(self.candidate_multiplier), int(n_target)))
        candidates: list[_LadderCandidate] = []
        seen: set[tuple[float, ...]] = set()
        for _, _, _, a, _tangent, local_scale in anchor_infos:
            direction = self._outward_direction(a, X_maj, maj_nn)
            if direction is None:
                continue
            max_depth = self.max_ladder_depth_factor * local_scale
            for frac in self.ladder_step_fracs:
                if len(candidates) >= max_raw:
                    break
                x = a + float(frac) * max_depth * direction
                if not np.all(np.isfinite(x)) or not _within_bounds(x, lo, hi):
                    continue
                dmin = float(min_nn.kneighbors(x.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0, 0])
                dmaj = float(maj_nn.kneighbors(x.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0, 0])
                ratio = dmaj / (dmin + EPS)
                tube_vec = x - a
                tube = float(np.linalg.norm(tube_vec - np.dot(tube_vec, direction) * direction))
                ok_majority = dmaj > dmin or (self.allow_minor_relaxation and dmaj >= 0.95 * dmin)
                ok = (
                    ok_majority
                    and ratio >= self.min_boundary_ratio
                    and ratio <= self.max_boundary_ratio
                    and dmin <= self.safety_factor * local_scale
                    and tube <= self.tube_radius_factor * local_scale
                )
                if not ok:
                    ok = (
                        ok_majority
                        and ratio >= self.min_boundary_ratio
                        and ratio <= 5.0
                        and dmin <= self.safety_factor * local_scale
                        and tube <= self.tube_radius_factor * local_scale
                    )
                if not ok:
                    continue
                d_real = real_nn.kneighbors(x.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0, 0]
                if d_real <= max(1e-8, 1e-6 * local_scale):
                    continue
                key = _candidate_key(x)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(_LadderCandidate(x=x, anchor=a, frac=float(frac), depth=float(np.linalg.norm(x - a)), d_min=dmin, d_maj=dmaj, ratio=ratio))
            if len(candidates) >= max_raw:
                break
        return candidates

    def _score_candidates(self, candidates: list[_LadderCandidate], selected_seed: np.ndarray) -> list[_LadderCandidate]:
        if not candidates:
            return []
        Xc = np.vstack([c.x for c in candidates])
        d_min = np.asarray([c.d_min for c in candidates], dtype=float)
        d_maj = np.asarray([c.d_maj for c in candidates], dtype=float)
        ratio = np.asarray([c.ratio for c in candidates], dtype=float)
        frac = np.asarray([c.frac for c in candidates], dtype=float)

        boundary_score = np.exp(-np.abs(ratio - self.boundary_preference))
        plausibility = 1.0 / (d_min + EPS)
        intrusion = 1.0 / (d_maj + EPS)
        depth_score = np.clip(1.0 - np.abs(frac - 0.40) / 0.40, 0.0, 1.0)

        coverage = np.zeros(len(candidates), dtype=float)
        if selected_seed is not None and len(selected_seed):
            seed = np.asarray(selected_seed, dtype=float)
            chunks = []
            for start in range(0, len(Xc), 1024):
                block = Xc[start:start + 1024]
                d = np.linalg.norm(block[:, None, :] - seed[None, :, :], axis=2)
                chunks.append(np.min(d, axis=1))
            coverage = np.concatenate(chunks) if chunks else coverage
        redundancy = 1.0 / (coverage + EPS)

        score = (
            self.boundary_weight * normalize01(boundary_score)
            + self.plausibility_weight * normalize01(plausibility)
            + self.coverage_weight * normalize01(coverage)
            + 0.15 * normalize01(depth_score)
            - self.intrusion_weight * normalize01(intrusion)
            - self.redundancy_weight * normalize01(redundancy)
        )
        rng = np.random.RandomState(0 if self.random_state is None else int(self.random_state) + 811)
        score = score + rng.uniform(0.0, 1e-9, size=len(score))
        order = np.argsort(score)[::-1]
        return [candidates[int(i)] for i in order]

    def fit_resample(self, X, y):
        X, y = to_numpy_xy(X, y)
        self.n_generated_ = 0
        self.n_base_smote_ = 0
        self.n_selected_ladder_ = 0
        self.warning_ = ""
        need = synthetic_needed(y, self.sampling_strategy)
        if need <= 0:
            return X.copy(), y.copy()
        X = np.nan_to_num(X.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        X_min = X[y == 1]
        X_maj = X[y == 0]
        if len(X_min) < 3 or len(X_maj) < 1:
            self.warning_ = "too few minority/majority samples for Fast Outward Ladder"
            return X.copy(), y.copy()
        rng = np.random.RandomState(self.random_state)
        n_base_target = int(round(self.base_smote_fraction * need))
        n_ladder_target = max(0, need - n_base_target)
        base_pool = _simple_smote(X_min, max(n_base_target, need), rng, k_neighbors=int(self.minority_neighbors), candidate_multiplier=int(self.candidate_multiplier))
        ladder_pool = self._ladder_candidates(X, X_min, X_maj, base_pool[:max(n_base_target, 1)], n_ladder_target, rng)
        ladder_selected = self._score_candidates(ladder_pool, base_pool[:n_base_target])[:n_ladder_target]
        ladder_syn = np.vstack([c.x for c in ladder_selected]) if ladder_selected else np.empty((0, X.shape[1]))
        base_need = need - len(ladder_syn)
        base_syn = base_pool[:base_need] if len(base_pool) else np.empty((0, X.shape[1]))
        if len(base_syn) < base_need:
            extra = _simple_smote(X_min, base_need - len(base_syn), rng, k_neighbors=int(self.minority_neighbors), candidate_multiplier=int(self.candidate_multiplier))
            base_syn = np.vstack([base_syn, extra]) if len(base_syn) and len(extra) else (extra if len(extra) else base_syn)

        synthetic = np.vstack([part for part in (base_syn, ladder_syn) if len(part)]) if (len(base_syn) or len(ladder_syn)) else np.empty((0, X.shape[1]))
        unique = []
        seen = {_candidate_key(r) for r in X}
        for row in synthetic:
            key = _candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
            if len(unique) >= need:
                break
        synthetic = np.vstack(unique) if unique else np.empty((0, X.shape[1]))
        self.n_generated_ = int(len(synthetic))
        self.n_base_smote_ = min(len(base_syn), self.n_generated_)
        self.n_selected_ladder_ = min(len(ladder_syn), self.n_generated_)
        if self.n_generated_ < need:
            self.warning_ = f"generated {self.n_generated_} of requested {need} synthetic points"
        if self.n_generated_ == 0:
            return X.copy(), y.copy()
        return np.vstack([X, synthetic]), np.concatenate([y, np.ones(self.n_generated_, dtype=y.dtype)])


OVERSAMPLER_ALIASES = {
    "none": "none",
    "native": "none",
    "no_sampling": "none",
    "random_over_sampler": "random_over_sampler",
    "ros": "random_over_sampler",
    "smote": "smote",
    "smote_tomek": "smote_tomek",
    "smotetomek": "smote_tomek",
    "kmeans_smote": "kmeans_smote",
    "adasyn": "adasyn",
    "borderline_smote": "borderline_smote",
    "deep_smote": "deep_smote",
    "deepsmote": "deep_smote",
    "mgvae": "mgvae",
    "fast_outward_ladder": "fast_outward_ladder",
    "fast_outward_ladder_smote": "fast_outward_ladder",
    "fol": "fast_outward_ladder",
}


def canonical_oversampler_name(name: str | None) -> str:
    key = "none" if name is None else str(name).strip().lower()
    if key not in OVERSAMPLER_ALIASES:
        raise ValueError(f"unknown oversampler {name!r}; choose from {sorted(set(OVERSAMPLER_ALIASES.values()))}")
    return OVERSAMPLER_ALIASES[key]


def make_oversampler(name: str | None, sampling_strategy: float = 1.0, random_state: int | None = None, **kwargs):
    name = canonical_oversampler_name(name)
    if name == "none":
        return NoneOversampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    if name in {"random_over_sampler", "smote", "smote_tomek", "kmeans_smote", "adasyn", "borderline_smote"}:
        return SafeImblearnOversampler(name, sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    if name == "deep_smote":
        return DeepSMOTEOversampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    if name == "mgvae":
        return MGVAEOversampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    if name == "fast_outward_ladder":
        return FastOutwardLadderOversampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    raise ValueError(f"unknown oversampler {name!r}")
