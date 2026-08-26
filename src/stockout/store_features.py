"""The per-store metadata features, and the two kinds of date feature inside them.

`store.csv` is one row per store, so joining it alone gives every day of a store's history
the same constant. What varies down the frame — and therefore what a model can use — is how
far *this row's date* sits from the competitor opening and from the promotion start.

Split out of `features.py` because it is the only Rossmann-shaped part of the matrix. The
product sources carry none of it, and `Source.has_store_metadata` is what says so.
"""

from __future__ import annotations

import pandas as pd

from .data import schemas as s
from .encoding import LABEL_VOCABULARIES, UNKNOWN

#: Month tokens as Rossmann spells them in `promo_interval`. "Sept" rather than "Sep"
#: is the source's spelling, not a typo.
MONTH_TOKENS: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sept", "Oct", "Nov", "Dec",
)

STORE_FEATURES: tuple[str, ...] = (
    s.STORE_TYPE,
    s.ASSORTMENT,
    s.COMPETITION_DISTANCE,
    "competition_open_age_months",
    s.PROMO2,
    "promo2_age_weeks",
    "is_promo2_month",
)

_MONTHS_PER_YEAR = 12
_DAYS_PER_WEEK = 7

def add_store_features(
    frame: pd.DataFrame,
    stores: pd.DataFrame | None,
    *,
    age_as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join per-entity metadata and turn its three date columns into ages.

    `store.csv` is one row per store, so joining it alone gives every day of a store's
    history the same constant. The useful features are how far *this row's date* is from
    the competitor opening and from the promotion start, which vary down the frame.

    `age_as_of` caps the date the **ages** are measured from, and exists for serving a
    date beyond the training window. The two kinds of date feature behave differently out
    there and the difference decides what is safe:

    The calendar features are cyclic. `day_of_week`, `month`, `week_of_year` for a day in
    2030 take values the fit has already seen thousands of times, so they extrapolate for
    free and must use the real date — capping them would answer about the wrong weekday.

    The ages are monotone. `competition_open_age_months` grows without bound, so a date
    years past the window supplies a value beyond every split point the trees were built
    from, and beyond the range any linear coefficient was fitted over. Left alone it is
    silent extrapolation on a feature carrying about a sixth of the deployed model's
    importance. Capped at the last fitted day it is the honest reading: *as far as this
    model can tell, this store's competitor has been open the longest it ever observed.*

    `is_promo2_month` is cyclic like the calendar, so it keeps the real date too.
    """
    frame = frame.copy()
    if stores is None:
        for column in STORE_FEATURES:
            frame[column] = _absent_value(column)
        return frame

    metadata = stores.copy()
    metadata[s.ENTITY] = metadata[s.ENTITY].astype("string")
    frame = frame.merge(metadata, on=s.ENTITY, how="left", suffixes=("", "_store"))

    dates = pd.to_datetime(frame[s.DATE])
    aged = dates if age_as_of is None else _capped(dates, pd.Timestamp(age_as_of))
    frame["competition_open_age_months"] = _age_in_months(
        aged, frame.get(s.COMPETITION_OPEN_YEAR), frame.get(s.COMPETITION_OPEN_MONTH)
    )
    frame["promo2_age_weeks"] = _age_in_weeks(
        aged, frame.get(s.PROMO2_SINCE_YEAR), frame.get(s.PROMO2_SINCE_WEEK)
    )
    frame["is_promo2_month"] = _is_promo2_month(dates, frame.get(s.PROMO_INTERVAL))

    for column in STORE_FEATURES:
        if column not in frame.columns:
            frame[column] = _absent_value(column)
    return frame

def _absent_value(column: str) -> str | int:
    return UNKNOWN if column in LABEL_VOCABULARIES else 0

def _capped(dates: pd.Series, ceiling: pd.Timestamp) -> pd.Series:
    """`dates` with anything after `ceiling` pulled back to it.

    `Series.clip(upper=...)` does this too, but its stubs do not admit a `Timestamp`, and
    silencing that would hide a real class of mistake at every other call site.
    """
    return dates.mask(dates > ceiling, ceiling)


def _age_in_months(
    dates: pd.Series, year: pd.Series | None, month: pd.Series | None
) -> pd.Series:
    """Months since the competitor opened. Negative before it did, 0 where unknown."""
    if year is None or month is None:
        return pd.Series(0.0, index=dates.index)
    months = (dates.dt.year - year) * _MONTHS_PER_YEAR + (dates.dt.month - month)
    return months.fillna(0).astype("float64")


def _age_in_weeks(
    dates: pd.Series, year: pd.Series | None, week: pd.Series | None
) -> pd.Series:
    """Weeks since the continuing promotion started, 0 where the store runs none."""
    if year is None or week is None:
        return pd.Series(0.0, index=dates.index)
    weeks = (dates.dt.year - year) * (365.25 / _DAYS_PER_WEEK) + (
        dates.dt.isocalendar().week - week
    )
    return weeks.fillna(0).astype("float64")


def _is_promo2_month(dates: pd.Series, interval: pd.Series | None) -> pd.Series:
    """Whether this row's month is one the store's continuing promotion repeats in.

    This is the feature `promo_interval` actually buys. On its own that column is three
    distinct values in the whole dataset and constant per store; crossed with the row's
    calendar it varies down the frame and says something a model can use.
    """
    if interval is None:
        return pd.Series(0, index=dates.index, dtype="int8")
    months = dates.dt.month.fillna(0).astype("int64").to_numpy()
    tokens = [
        MONTH_TOKENS[month - 1] if 1 <= month <= _MONTHS_PER_YEAR else ""
        for month in months
    ]
    text = interval.astype("string").fillna("")
    flags = [
        1 if token and token in row else 0
        for token, row in zip(tokens, text, strict=True)
    ]
    return pd.Series(flags, index=dates.index, dtype="int8")
