from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .utils import class_counts

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "bestangle25"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"

BESTANGLE25_DATASETS = [
    "blood-transfusion",
    "diabetes",
    "ecoli",
    "haberman",
    "ionosphere",
    "openml-1068-pc1",
    "sonar",
    "keel-abalone9-18",
    "tic-tac-toe",
    "keel-ecoli2",
    "keel-ecoli3",
    "keel-glass2",
    "keel-glass6",
    "keel-yeast1",
    "keel-yeast3",
    "keel-yeast5",
    "openml-1487-ozone-level-8hr",
    "openml-378-ipums_la_99-small",
    "openml-40664-car-evaluation",
    "openml-40691-wine-quality-red",
    "credit-g",
    "keel-page-blocks0",
    "keel-vehicle1",
    "vehicle",
    "keel-flare-F",
]

PRIMARY_NUMERIC18_DATASETS = [
    "blood-transfusion",
    "diabetes",
    "ecoli",
    "ionosphere",
    "openml-1068-pc1",
    "sonar",
    "keel-ecoli2",
    "keel-ecoli3",
    "keel-glass2",
    "keel-glass6",
    "keel-yeast1",
    "keel-yeast3",
    "keel-yeast5",
    "openml-1487-ozone-level-8hr",
    "openml-40691-wine-quality-red",
    "keel-page-blocks0",
    "keel-vehicle1",
    "vehicle",
]

AUXILIARY_CATEGORY7_DATASETS = [
    "credit-g",
    "haberman",
    "keel-abalone9-18",
    "keel-flare-F",
    "openml-378-ipums_la_99-small",
    "openml-40664-car-evaluation",
    "tic-tac-toe",
]

# Compatibility alias for callers of the first repository release.
FAST25_DATASETS = BESTANGLE25_DATASETS


@dataclass
class DatasetRecord:
    X: np.ndarray
    y: np.ndarray
    name: str
    source: str
    metadata: dict


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _encode_frame(
    X_df: pd.DataFrame,
    *,
    allow_categorical: bool = False,
) -> np.ndarray:
    categorical = [
        str(column)
        for column in X_df.columns
        if not pd.api.types.is_numeric_dtype(X_df[column])
    ]
    if categorical and not allow_categorical:
        raise ValueError(
            "categorical predictors are not accepted by the numerical "
            "PSTE/FOL protocol because interpolation over ordinal codes is "
            "not semantically valid. Encode them with a category-safe method, "
            "or pass allow_categorical=True only for an explicitly auxiliary "
            f"study. Categorical columns: {categorical}"
        )
    encoded = pd.DataFrame(index=X_df.index)
    for col in X_df.columns:
        s = X_df[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            cat = pd.Categorical(s)
            vals = pd.Series(cat.codes, index=s.index, dtype=float)
            vals[vals < 0] = np.nan
        encoded[str(col)] = vals.replace([np.inf, -np.inf], np.nan)
    keep = []
    for col in encoded.columns:
        values = encoded[col]
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    if not keep:
        raise ValueError("no usable feature columns after encoding")
    return encoded[keep].to_numpy(dtype=float)


def _binary_labels(y_raw) -> tuple[np.ndarray, dict]:
    y_series = pd.Series(y_raw).reset_index(drop=True)
    valid = ~y_series.isna()
    y_series = y_series.loc[valid].reset_index(drop=True)
    labels, uniques = pd.factorize(y_series, sort=True)
    if len(uniques) < 2:
        raise ValueError("target has fewer than two classes")
    counts = pd.Series(labels).value_counts().sort_index()
    if len(uniques) == 2:
        positive_code = int(counts.idxmin())
    else:
        eligible = counts[counts >= 10]
        positive_code = int((eligible if len(eligible) else counts).idxmin())
    y = (labels == positive_code).astype(int)
    if len(uniques) == 2 and np.sum(y == 1) > np.sum(y == 0):
        y = 1 - y
    meta = {
        "original_classes": [str(u) for u in uniques],
        "positive_class": str(uniques[positive_code]),
        "original_counts": {str(uniques[int(i)]): int(c) for i, c in counts.items()},
    }
    return y.astype(int), meta


def load_csv_dataset(
    path: str | Path,
    target: str | None = None,
    name: str | None = None,
    *,
    allow_categorical: bool = False,
) -> DatasetRecord:
    path = Path(path)
    df = pd.read_csv(path)
    if target is None:
        target = df.columns[-1]
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in {path}")
    y_raw = df[target]
    X_df = df.drop(columns=[target])
    valid = ~pd.Series(y_raw).isna().reset_index(drop=True)
    X_df = X_df.reset_index(drop=True).loc[valid].reset_index(drop=True)
    y_raw = pd.Series(y_raw).reset_index(drop=True).loc[valid].reset_index(drop=True)
    X = _encode_frame(X_df, allow_categorical=allow_categorical)
    y, meta = _binary_labels(y_raw)
    meta.update({"file": str(path), "target": target, "class_counts": class_counts(y)})
    return DatasetRecord(X=X, y=y, name=name or path.stem, source="csv", metadata=meta)


def load_dataset(
    name_or_path: str,
    data_dir: str | Path = DATA_DIR,
    target: str | None = None,
    *,
    allow_categorical: bool = False,
) -> DatasetRecord:
    path = Path(name_or_path)
    if path.exists() and path.suffix.lower() == ".csv":
        return load_csv_dataset(
            path,
            target=target,
            allow_categorical=allow_categorical,
        )
    name = str(name_or_path)
    p = Path(data_dir) / f"{name}.joblib"
    if not p.exists():
        available = ", ".join(BESTANGLE25_DATASETS)
        raise KeyError(f"unknown packaged dataset {name!r}. Use one of: {available}; or pass a CSV path.")
    payload = joblib.load(p)
    return DatasetRecord(
        X=np.asarray(payload["X"], dtype=float),
        y=np.asarray(payload["y"], dtype=int),
        name=str(payload.get("name", name)),
        source=str(payload.get("source", "packaged")),
        metadata=dict(payload.get("metadata", {})),
    )


def resolve_dataset_names(names: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in names:
        key = str(raw).strip()
        low = key.lower()
        if low in {
            "paper",
            "current",
            "primary",
            "numeric18",
            "kbs18",
        }:
            out.extend(PRIMARY_NUMERIC18_DATASETS)
        elif low in {"bestangle25", "fast25", "all", "all25"}:
            out.extend(BESTANGLE25_DATASETS)
        elif low in {"auxiliary", "category7"}:
            out.extend(AUXILIARY_CATEGORY7_DATASETS)
        else:
            out.append(key)
    seen = set()
    deduped = []
    for name in out:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def load_datasets(
    names: Sequence[str],
    data_dir: str | Path = DATA_DIR,
    target: str | None = None,
    *,
    allow_categorical: bool = False,
) -> list[DatasetRecord]:
    return [
        load_dataset(
            name,
            data_dir=data_dir,
            target=target,
            allow_categorical=allow_categorical,
        )
        for name in resolve_dataset_names(names)
    ]
