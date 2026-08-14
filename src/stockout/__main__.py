"""Enables `python -m stockout`, so the package runs without being on PATH."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
