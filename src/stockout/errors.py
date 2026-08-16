"""Domain exceptions, so the CLI can print `error: ...` instead of a traceback.

`LeakageError` is the one that matters. It is raised for a *programming* mistake —
asking for a feature that could not exist at the forecast origin — and it is raised
eagerly, at feature-build time, rather than showing up as a suspiciously good score
three steps later.
"""

from __future__ import annotations


class StockoutError(Exception):
    """Base for every error this package raises deliberately."""


class DownloadError(StockoutError):
    """The Kaggle archive could not be fetched or unpacked."""


class SchemaError(StockoutError):
    """A frame violated an assumption recorded in `stockout.data.schemas`."""


class LeakageError(StockoutError):
    """A feature or split would have used information from the future."""


class BacktestError(StockoutError):
    """A backtest could not be laid out — usually too little history for the folds asked for."""


class MissingDependencyError(StockoutError):
    """An optional extra is needed for this path and is not installed.

    Separate from `ImportError` on purpose: the CLI already turns a `StockoutError` into
    one readable line, and "install the extra" is a user instruction, not a stack trace.
    """
