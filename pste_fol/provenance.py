from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "imbalanced-learn",
    "imbalanced-learn-extra",
    "joblib",
    "torch",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def runtime_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        result["cuda_available"] = cuda_available
        result["torch_cuda_runtime"] = str(torch.version.cuda or "")
        result["cuda_device"] = (
            str(torch.cuda.get_device_name(0)) if cuda_available else None
        )
    except (ImportError, RuntimeError):
        result["cuda_available"] = False
        result["torch_cuda_runtime"] = None
        result["cuda_device"] = None
    return result


def git_state(root: str | Path) -> dict[str, object]:
    root = Path(root)
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def artifact_record(path: str | Path) -> dict[str, object]:
    path = Path(path)
    return {
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
