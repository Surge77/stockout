"""Retail demand forecasting scored by the decision it drives.

    from stockout.data.synth import make_sales
    from stockout.evaluate.backtest import backtest
    from stockout.models.baselines import SeasonalNaive

    frame = make_sales(n_stores=8, days=730)
    results = backtest(frame, SeasonalNaive, n_folds=5, horizon=42)

Submodules are deliberately not imported here. Importing the package should not
pull in matplotlib or kaggle; `stockout describe` has no business opening a
plotting backend.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# Read from the installed distribution rather than typed here. Hand-maintained twice, this
# said 0.1.0 while pyproject said 0.3.0 through two releases — the failure is silent,
# because nothing imports a version to check that it is wrong. One source now.
try:
    __version__ = version("stockout")
except PackageNotFoundError:  # pragma: no cover - a source tree on PYTHONPATH, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
