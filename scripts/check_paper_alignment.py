#!/usr/bin/env python3
"""Backward-compatible alias for scripts/check_reference.py."""

from check_reference import main


if __name__ == "__main__":
    raise SystemExit(main())
