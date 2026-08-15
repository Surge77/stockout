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

__version__ = "0.1.0"

__all__ = ["__version__"]
