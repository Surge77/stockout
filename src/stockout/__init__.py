"""stockout — Retail Sales and Product Demand Prediction.

Predicts store-day sales, and classifies each store-day as Low, Medium or High demand.

    from stockout.datasets import SALARY
    from stockout.evaluate import compare

    print(compare(SALARY, "regression"))

Submodules are deliberately not imported here. Importing the package should not pull in
matplotlib; `stockout describe` has no business opening a plotting backend.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("stockout")
except PackageNotFoundError:  # pragma: no cover - a source tree, not an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
