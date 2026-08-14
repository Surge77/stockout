"""Assertions about a sales frame, run before anything is allowed to model it.

These are cheap and they fail loudly. The alternative — discovering that a store has
two rows for one date after a backtest has produced a plausible-looking number — costs
an afternoon.
"""

from __future__ import annotations

import pandas as pd

from ..errors import SchemaError
from . import schemas as s


def validate_sales(frame: pd.DataFrame) -> None:
    """Raise `SchemaError` unless `frame` satisfies every recorded assumption."""
    _require_columns(frame)
    _require_datetime(frame)
    _require_unique_keys(frame)
    _require_binary_open(frame)
    _require_closed_days_are_zero(frame)
    _require_non_negative_sales(frame)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in s.REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing required columns: {', '.join(missing)}")


def _require_datetime(frame: pd.DataFrame) -> None:
    if not pd.api.types.is_datetime64_any_dtype(frame[s.DATE]):
        raise SchemaError(f"{s.DATE!r} must be datetime64, got {frame[s.DATE].dtype}")


def _require_unique_keys(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(subset=list(s.KEY_COLUMNS))
    if bool(duplicated.any()):
        example = frame.loc[duplicated, list(s.KEY_COLUMNS)].iloc[0].to_dict()
        raise SchemaError(
            f"{int(duplicated.sum())} duplicate (store, date) rows; first example: {example}"
        )


def _require_binary_open(frame: pd.DataFrame) -> None:
    bad = set(frame[s.OPEN].unique()) - {0, 1}
    if bad:
        raise SchemaError(f"{s.OPEN!r} must be 0 or 1, found {sorted(bad)}")


def _require_closed_days_are_zero(frame: pd.DataFrame) -> None:
    """A closed store cannot take money.

    This invariant is what lets the backtest exclude closed days: they are perfectly
    predictable zeros, and leaving them in flatters any per-row average such as MAE.
    """
    offending = frame[(frame[s.OPEN] == 0) & (frame[s.SALES] != 0)]
    if not offending.empty:
        raise SchemaError(
            f"{len(offending)} rows have open=0 but non-zero sales; "
            f"first on {offending.iloc[0][s.DATE].date()} at store {offending.iloc[0][s.STORE]}"
        )


def _require_non_negative_sales(frame: pd.DataFrame) -> None:
    if bool((frame[s.SALES] < 0).any()):
        raise SchemaError("negative sales are not returns; they are a parsing bug")


def null_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column null count and share, worst first. Reported by `stockout describe`."""
    nulls = frame.isna().sum()
    profile = pd.DataFrame(
        {
            "column": nulls.index,
            "nulls": nulls.to_numpy(),
            "pct": (nulls.to_numpy() / max(len(frame), 1) * 100).round(2),
        }
    )
    return profile.sort_values(["nulls", "column"], ascending=[False, True]).reset_index(drop=True)


def calendar_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Stores whose date range has holes, and how many days are missing.

    Rossmann has these: a subset of stores closes for refurbishment and simply stops
    appearing for months. That produces a level shift on reopening which looks like
    signal and is not. Quantify it before trusting any store-level result.
    """
    rows = []
    for store, group in frame.groupby(s.STORE, observed=True):
        span = (group[s.DATE].max() - group[s.DATE].min()).days + 1
        missing = span - len(group)
        if missing > 0:
            rows.append({s.STORE: store, "expected_days": span, "missing_days": missing})
    return pd.DataFrame(rows, columns=[s.STORE, "expected_days", "missing_days"])
