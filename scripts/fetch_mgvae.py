#!/usr/bin/env python3
"""Fetch the exact upstream MGVAE revision used by the locked benchmark."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/Aiqz/MGVAE.git"
COMMIT = "cad386bd2b3a90f6b740cbdf5f0cec8834102ea5"


def run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        default=str(ROOT / "external" / "MGVAE"),
    )
    args = parser.parse_args(argv)
    destination = Path(args.destination).expanduser().resolve()

    if destination.exists():
        if not (destination / ".git").is_dir():
            raise SystemExit(
                f"{destination} exists but is not a Git checkout; refusing to replace it"
            )
        if run("git", "status", "--porcelain", cwd=destination):
            raise SystemExit(
                f"{destination} has local changes; refusing to alter it"
            )
        remote = run("git", "remote", "get-url", "origin", cwd=destination)
        if remote.rstrip("/") != REPOSITORY.rstrip("/"):
            raise SystemExit(
                f"{destination} origin is {remote!r}, expected {REPOSITORY!r}"
            )
        run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY,
            str(destination),
        )
        run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=destination)

    run("git", "checkout", "--detach", COMMIT, cwd=destination)
    actual = run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != COMMIT:
        raise SystemExit(f"checkout verification failed: {actual} != {COMMIT}")
    required = destination / "models" / "models_mgvae" / "mgvae_tabular.py"
    if not required.is_file():
        raise SystemExit(f"required upstream source is missing: {required}")
    print(f"MGVAE ready at {destination}")
    print(f"commit {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
