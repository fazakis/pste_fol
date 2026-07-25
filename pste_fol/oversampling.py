from __future__ import annotations

import warnings
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
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
        if self.method in {"geometric_smote", "gsmote"}:
            from imblearn_extra.gsmote import GeometricSMOTE

            return GeometricSMOTE(k_neighbors=k, **common)
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


MGVAE_REPOSITORY = "https://github.com/Aiqz/MGVAE.git"
MGVAE_COMMIT = "cad386bd2b3a90f6b740cbdf5f0cec8834102ea5"
_MGVAE_TABULAR_CLASS = None


def _mgvae_repository_root() -> Path:
    import os

    configured = os.environ.get("PSTE_FOL_MGVAE_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else Path(__file__).resolve().parents[1] / "external" / "MGVAE"
    )
    required = root / "models" / "models_mgvae" / "mgvae_tabular.py"
    if not required.exists():
        raise RuntimeError(
            "The pinned original MGVAE implementation is missing. Run "
            "`python scripts/fetch_mgvae.py`, or set PSTE_FOL_MGVAE_ROOT "
            f"to a checkout of {MGVAE_REPOSITORY} at {MGVAE_COMMIT}."
        )
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"{root} is not a verifiable Git checkout of the pinned MGVAE "
            "source; run `python scripts/fetch_mgvae.py`"
        ) from exc
    if actual_commit != MGVAE_COMMIT:
        raise RuntimeError(
            f"MGVAE checkout is at {actual_commit}, expected {MGVAE_COMMIT}; "
            "run `python scripts/fetch_mgvae.py`"
        )
    return root


def _load_original_mgvae_tabular_class():
    """Load the pinned upstream MGVAE_Tabular class without package side effects."""

    global _MGVAE_TABULAR_CLASS
    if _MGVAE_TABULAR_CLASS is not None:
        return _MGVAE_TABULAR_CLASS

    import importlib.util
    import sys
    import types

    repository = _mgvae_repository_root()
    root_name = "_pste_fol_original_mgvae"
    models_name = f"{root_name}.models"
    mgvae_package_name = f"{models_name}.models_mgvae"
    for name, package_path in (
        (root_name, repository),
        (models_name, repository / "models"),
        (mgvae_package_name, repository / "models" / "models_mgvae"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(package_path)]
            sys.modules[name] = package

    utils_name = f"{models_name}.utils"
    if utils_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            utils_name,
            repository / "models" / "utils.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load pinned MGVAE models/utils.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[utils_name] = module
        spec.loader.exec_module(module)

    module_name = f"{mgvae_package_name}.mgvae_tabular"
    spec = importlib.util.spec_from_file_location(
        module_name,
        repository / "models" / "models_mgvae" / "mgvae_tabular.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned MGVAE_Tabular implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MGVAE_TABULAR_CLASS = module.MGVAE_Tabular
    return _MGVAE_TABULAR_CLASS


def _fit_feature_minmax_to_tanh(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    low = np.nanmin(X, axis=0)
    high = np.nanmax(X, axis=0)
    span = high - low
    safe_span = np.where(span > 1e-12, span, 1.0)
    scaled = 2.0 * (X - low) / safe_span - 1.0
    scaled[:, span <= 1e-12] = 0.0
    return scaled.astype(np.float32), low.astype(float), safe_span.astype(float)


def _invert_feature_minmax_from_tanh(
    X_scaled: np.ndarray,
    low: np.ndarray,
    safe_span: np.ndarray,
) -> np.ndarray:
    X_scaled = np.clip(np.asarray(X_scaled, dtype=float), -1.0, 1.0)
    return ((X_scaled + 1.0) * 0.5) * safe_span + low


class MGVAEOversampler:
    """Exact wrapper around the pinned Aiqz/MGVAE tabular implementation.

    The locked primary benchmark uses 200 epochs, latent dimension 10, hidden
    size 300, 700 mixture components, batch size 100, learning rate 0.005, and
    KLD weight 1.0. The upstream source is fetched separately because its
    repository does not publish a redistribution license.
    """

    method_name = "MGVAE"
    implementation_reference_ = MGVAE_REPOSITORY
    original_repo_commit_ = MGVAE_COMMIT

    def __init__(
        self,
        sampling_strategy: float = 1.0,
        random_state: int | None = 42,
        *,
        latent_dim: int = 10,
        hidden_size: int = 300,
        number_components: int = 700,
        epochs: int = 200,
        batch_size: int = 100,
        lr: float = 0.005,
        weight_decay: float = 0.0,
        kld_weight: float = 1.0,
        device: str = "auto",
        torch_threads: int | None = 1,
        **kwargs,
    ):
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.number_components = int(number_components)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.kld_weight = float(kld_weight)
        self.device = str(device)
        self.torch_threads = torch_threads
        self.extra_params = dict(kwargs)
        self.n_generated_ = 0
        self.warning_ = ""

    def _device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit_resample(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X, y = to_numpy_xy(X, y)
        self.n_generated_ = 0
        self.warning_ = ""
        needed = synthetic_needed(y, self.sampling_strategy)
        if needed <= 0:
            return X.copy(), y.copy()
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError("original MGVAE requires both binary classes")

        seed = 0 if self.random_state is None else int(self.random_state)
        if self.torch_threads is not None:
            torch.set_num_threads(max(1, int(self.torch_threads)))
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = self._device()

        scaled, self.feature_min_, self.feature_safe_span_ = (
            _fit_feature_minmax_to_tanh(X)
        )
        majority = torch.tensor(scaled[y == 0], dtype=torch.float32)
        labels = torch.zeros(len(majority), dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loader = DataLoader(
            TensorDataset(majority, labels),
            batch_size=max(2, min(self.batch_size, len(majority))),
            shuffle=True,
            num_workers=0,
            generator=generator,
        )

        model_class = _load_original_mgvae_tabular_class()
        model = model_class(
            input_dim=X.shape[1],
            latent_dim=self.latent_dim,
            hidden_size=self.hidden_size,
            number_components=self.number_components,
            dataset_loader=loader,
            cur_device=str(device),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        losses = []
        model.train()
        for _epoch in range(max(1, self.epochs)):
            epoch_loss = 0.0
            for real, batch_labels in loader:
                optimizer.zero_grad()
                real = real.to(device)
                batch_labels = batch_labels.to(device)
                results = model(real, labels=batch_labels)
                loss = model.loss_function(
                    *results,
                    M_N=self.kld_weight,
                    optimizer_idx=0,
                    batch_idx=0,
                )["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item()) * int(real.size(0))
            losses.append(epoch_loss / max(1, len(loader.dataset)))
        self.training_loss_ = float(losses[-1]) if losses else np.nan

        model.eval()
        with torch.no_grad():
            generated_scaled = (
                model.index_sample(
                    list(range(int(needed))),
                    current_device=str(device),
                )
                .detach()
                .cpu()
                .numpy()
            )
        generated = _invert_feature_minmax_from_tanh(
            generated_scaled,
            self.feature_min_,
            self.feature_safe_span_,
        )
        if not np.all(np.isfinite(generated)):
            raise ValueError("original MGVAE generated non-finite values")
        generated_labels = np.ones(len(generated), dtype=y.dtype)
        self.n_generated_ = int(len(generated))
        return (
            np.vstack([X, np.asarray(generated, dtype=float)]),
            np.concatenate([y, generated_labels]),
        )


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
        random_state: int | None = 42,
        *,
        k_neighbors: int = 5,
        base_smote_fraction: float = 0.40,
        ladder_step_fracs: Iterable[float] = (0.12, 0.24, 0.38, 0.54),
        max_ladder_depth_factor: float = 1.05,
        candidate_multiplier: int = 6,
        max_candidates: int = 8000,
        minority_neighbors: int = 7,
        majority_neighbors: int = 8,
        inward_target_mode: str = "nearest_majority_centroid",
        boundary_preference: float = 1.35,
        min_boundary_ratio: float = 1.02,
        max_boundary_ratio: float = 5.0,
        safety_factor: float = 2.40,
        tube_radius_factor: float = 0.90,
        boundary_weight: float = 0.30,
        plausibility_weight: float = 0.28,
        coverage_weight: float = 0.45,
        intrusion_weight: float = 0.42,
        redundancy_weight: float = 0.06,
        allow_minor_relaxation: bool = False,
        **kwargs,
    ):
        self.sampling_strategy = float(sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = int(k_neighbors)
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
        self.n_ladder_candidates_ = 0
        self.n_selected_ladder_ = 0
        self.n_rejected_ladder_ = 0
        self.mean_ladder_depth_ = np.nan
        self.mean_boundary_ratio_ = np.nan
        self.warning_ = ""
        self.selected_base_points_ = np.empty((0, 0))
        self.selected_ladder_points_ = np.empty((0, 0))
        self.ladder_anchor_points_ = np.empty((0, 0))
        self.ladder_segments_: list[tuple[list[float], list[float]]] = []

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
                    self.n_rejected_ladder_ += 1
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
                    self.n_rejected_ladder_ += 1
                    continue
                d_real = real_nn.kneighbors(x.reshape(1, -1), n_neighbors=1, return_distance=True)[0][0, 0]
                if d_real <= max(1e-8, 1e-6 * local_scale):
                    self.n_rejected_ladder_ += 1
                    continue
                key = _candidate_key(x)
                if key in seen:
                    continue
                seen.add(key)
                self.n_ladder_candidates_ += 1
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
        self.n_ladder_candidates_ = 0
        self.n_selected_ladder_ = 0
        self.n_rejected_ladder_ = 0
        self.mean_ladder_depth_ = np.nan
        self.mean_boundary_ratio_ = np.nan
        self.warning_ = ""
        self.selected_base_points_ = np.empty((0, X.shape[1] if np.asarray(X).ndim == 2 else 0))
        self.selected_ladder_points_ = np.empty_like(self.selected_base_points_)
        self.ladder_anchor_points_ = np.empty_like(self.selected_base_points_)
        self.ladder_segments_ = []
        need = synthetic_needed(y, self.sampling_strategy)
        if need <= 0:
            return X.copy(), y.copy()
        X = np.nan_to_num(X.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        X_min = X[y == 1]
        X_maj = X[y == 0]
        if len(X_min) < 3 or len(X_maj) < 1:
            self.warning_ = "too few minority/majority samples for Fast Outward Ladder"
            return X.copy(), y.copy()
        rng = np.random.RandomState(0 if self.random_state is None else int(self.random_state))
        n_base_target = int(round(self.base_smote_fraction * need))
        n_ladder_target = max(0, need - n_base_target)
        base_pool = _simple_smote(
            X_min,
            max(n_base_target, need),
            rng,
            k_neighbors=int(self.k_neighbors),
            candidate_multiplier=int(self.candidate_multiplier),
        )
        ladder_pool = self._ladder_candidates(X, X_min, X_maj, base_pool[:max(n_base_target, 1)], n_ladder_target, rng)
        ladder_selected = self._score_candidates(ladder_pool, base_pool[:n_base_target])[:n_ladder_target]
        ladder_syn = np.vstack([c.x for c in ladder_selected]) if ladder_selected else np.empty((0, X.shape[1]))
        base_need = need - len(ladder_syn)
        base_syn = base_pool[:base_need] if len(base_pool) else np.empty((0, X.shape[1]))
        if len(base_syn) < base_need:
            extra = _simple_smote(
                X_min,
                base_need - len(base_syn),
                rng,
                k_neighbors=int(self.k_neighbors),
                candidate_multiplier=int(self.candidate_multiplier),
            )
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
        self.n_selected_ladder_ = int(len(ladder_syn))
        self.selected_base_points_ = np.asarray(
            base_syn[:max(0, need - len(ladder_syn))],
            dtype=float,
        )
        self.selected_ladder_points_ = np.asarray(ladder_syn, dtype=float)
        self.ladder_anchor_points_ = (
            np.vstack([candidate.anchor for candidate in ladder_selected])
            if ladder_selected
            else np.empty((0, X.shape[1]), dtype=float)
        )
        self.ladder_segments_ = [
            (candidate.anchor.tolist(), candidate.x.tolist())
            for candidate in ladder_selected
        ]
        if ladder_selected:
            self.mean_ladder_depth_ = float(
                np.mean([candidate.depth for candidate in ladder_selected])
            )
            self.mean_boundary_ratio_ = float(
                np.mean([candidate.ratio for candidate in ladder_selected])
            )
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
    "geometric_smote": "geometric_smote",
    "gsmote": "geometric_smote",
    "adasyn": "adasyn",
    "borderline_smote": "borderline_smote",
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
    if name in {
        "random_over_sampler",
        "smote",
        "smote_tomek",
        "kmeans_smote",
        "geometric_smote",
        "adasyn",
        "borderline_smote",
    }:
        return SafeImblearnOversampler(name, sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    if name == "mgvae":
        return MGVAEOversampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    if name == "fast_outward_ladder":
        return FastOutwardLadderOversampler(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)
    raise ValueError(f"unknown oversampler {name!r}")
