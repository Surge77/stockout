"""Turn store metadata into features that vary with the row's date.

`store.csv` is one row per store, so joining it gives every day of a store's history the
same constant. Three of its columns are not really constants at all — they are *dates*,
and the useful feature is how far the row is from them:

- `competition_open_since_{month,year}` → how long a competitor has been open on this day
- `promo2_since_{week,year}` → how long the continuing promotion has been running
- `promo_interval` → whether *this* row's month is one the promotion repeats in

That last one matters more than it looks. `promo_interval` has three distinct values in
the whole dataset, so on its own it is a three-level category wearing a text costume, and
a Tf-idf of it is a scaled one-hot that cannot move any metric — it is constant for a
store and constant in time. Crossed with the calendar it stops being constant in time,
and becomes the only feature here that says "this store is on promotion *today*".

Everything in this module is future-known: a promotion calendar for six weeks' time
already exists in a planning system. Nothing here reads `sales`, so none of it can leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import schemas as s

COMPETITION_OPEN_MONTHS = "competition_open_months"
PROMO2_ACTIVE_WEEKS = "promo2_active_weeks"
IS_PROMO2_MONTH = "is_promo2_month"

STORE_FEATURE_COLUMNS: tuple[str, ...] = (
    COMPETITION_OPEN_MONTHS,
    PROMO2_ACTIVE_WEEKS,
    IS_PROMO2_MONTH,
)

#: Rossmann's own month abbreviations, mapped to month numbers. September is spelled
#: `Sept` in the source file, not `Sep`. Both are accepted rather than only the one
#: currently shipped, because this mapping breaking is silent — an unmatched month makes
#: `is_promo2_month` quietly zero everywhere and the feature simply stops working.
_MONTH_TOKENS: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip

_MONTHS_PER_YEAR = 12
_DAYS_PER_WEEK = 7


def add_store_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the three date-relative store features. Pure; does not mutate input.

    Silently skips any feature whose source columns are absent, so a frame that never
    had `store.csv` joined onto it passes through unchanged rather than raising. The
    alternative — requiring the join — would make every baseline test carry metadata it
    does not use.
    """
    out = frame.copy()
    if s.DATE not in out.columns:
        return out

    if _has(out, s.COMPETITION_OPEN_MONTH, s.COMPETITION_OPEN_YEAR):
        out[COMPETITION_OPEN_MONTHS] = _competition_open_months(out)
    if _has(out, s.PROMO2_SINCE_WEEK, s.PROMO2_SINCE_YEAR):
        out[PROMO2_ACTIVE_WEEKS] = _promo2_active_weeks(out)
    if s.PROMO_INTERVAL in out.columns:
        out[IS_PROMO2_MONTH] = _is_promo2_month(out)
    return out


def _has(frame: pd.DataFrame, *columns: str) -> bool:
    return all(column in frame.columns for column in columns)


def _competition_open_months(frame: pd.DataFrame) -> pd.Series:
    """Whole months between the competitor opening and this row's date.

    Negative before the competitor opened, which is kept rather than clipped to zero:
    "opens in four months" and "opened four months ago" are different situations, and a
    tree can split on the sign. NaN where the opening date is unknown — imputing it here
    would hide a decision that `preprocess.py` makes in the open.
    """
    months = (
        (frame[s.DATE].dt.year - frame[s.COMPETITION_OPEN_YEAR]) * _MONTHS_PER_YEAR
        + frame[s.DATE].dt.month
        - frame[s.COMPETITION_OPEN_MONTH]
    )
    return months.astype("float64")


def _promo2_active_weeks(frame: pd.DataFrame) -> pd.Series:
    """Whole weeks the continuing promotion has been running on this row's date.

    `promo2_since_week` is an **ISO** week number, so the start date is resolved with
    `%G-W%V-%u` (ISO year, ISO week, ISO weekday) and not with `%Y%W`. The two
    conventions disagree by up to a week at the turn of the year, which is exactly where
    the December and January rows live.

    Zero, not NaN, for a store that does not run the promotion: the answer is known and
    it is "none". NaN would mean "unknown" and would send the row to an imputer that
    would invent a duration.
    """
    started = _iso_week_start(frame[s.PROMO2_SINCE_YEAR], frame[s.PROMO2_SINCE_WEEK])
    weeks = (frame[s.DATE] - started).dt.days / _DAYS_PER_WEEK
    return weeks.fillna(0.0).clip(lower=0.0).astype("float64")


def _iso_week_start(year: pd.Series, week: pd.Series) -> pd.Series:
    """Monday of the given ISO year and ISO week. NaT wherever either input is missing."""
    known = year.notna() & week.notna()
    stamps = pd.Series(pd.NaT, index=year.index, dtype="datetime64[ns]")
    if not bool(known.any()):
        return stamps

    text = (
        year[known].astype("int64").astype(str)
        + "-W"
        + week[known].astype("int64").astype(str).str.zfill(2)
        + "-1"
    )
    stamps.loc[known] = pd.to_datetime(text, format="%G-W%V-%u")
    return stamps


def _is_promo2_month(frame: pd.DataFrame) -> pd.Series:
    """1 where this row's month is one the continuing promotion repeats in.

    The join between a store-constant text column and the row's own calendar, and the
    only thing that makes `promo_interval` predictive rather than decorative.
    """
    months = frame[s.DATE].dt.month.to_numpy()
    intervals = frame[s.PROMO_INTERVAL].fillna("").astype(str)

    # One parse per distinct calendar, not one per row. There are three of them in the
    # whole dataset and roughly a million rows, so the map is the difference between a
    # vectorised comparison and a million string splits.
    lookup = {value: _months_in(value) for value in intervals.unique()}
    active = intervals.map(lookup)

    flags = [month in allowed for month, allowed in zip(months, active, strict=True)]
    return pd.Series(np.array(flags, dtype="int8"), index=frame.index)


def _months_in(interval: str) -> frozenset[int]:
    """`"Jan,Apr,Jul,Oct"` -> `{1, 4, 7, 10}`. An unrecognised token is dropped."""
    tokens = (token.strip().lower() for token in interval.split(","))
    return frozenset(_MONTH_TOKENS[token] for token in tokens if token in _MONTH_TOKENS)
