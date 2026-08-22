"""One exception type, so the CLI prints `error: ...` instead of a traceback."""

from __future__ import annotations


class StockoutError(Exception):
    """Every error this package raises deliberately."""


class SchemaError(StockoutError):
    """A frame violated an assumption recorded in `stockout.data.schemas`."""
