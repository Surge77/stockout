"""Build the model matrix from things that are knowable before the day happens.

**No lags, no rolling means.** Every feature here comes from the calendar, from the
store's own metadata, or from a flag that is decided in advance — a promotion calendar for
next month is already sitting in a planning system. Nothing is derived by shifting the
sales column, so there is nothing for a shuffled split to leak through and
`train_test_split` stays defensible.

What that costs, stated rather than hidden: this answers *what sales should we expect at
this store on this kind of day*, not *forecast the next six weeks*. Two adjacent days at
one store are near-duplicates, so a shuffled split still flatters the score a little.
Much smaller than a lag leak, not zero.

Categoricals go through `LabelEncoder` with a **fixed** vocabulary rather than one learned
from the fold, so a store type the training rows never contained cannot raise inside a web
request. Label encoding implies an ordering that does not exist; trees cope with that and
the linear models less so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data import schemas as s

#: Month tokens as Rossmann spells them in `promo_interval`. "Sept" rather than "Sep"
#: is the source's spelling, not a typo.
MONTH_TOKENS: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sept", "Oct", "Nov", "Dec",
)

#: Anything outside a column's known vocabulary becomes this. Present in every
#: vocabulary below, so `LabelEncoder.transform` always has a code for it.
UNKNOWN = "unknown"

#: Fixed up front rather than learned. These are closed sets: Rossmann has four store
#: types, three assortments and four holiday codes, and all of them are known before any
#: data is read.
LABEL_VOCABULARIES: dict[str, tuple[str, ...]] = {
    s.STORE_TYPE: ("a", "b", "c", "d", UNKNOWN),
    s.ASSORTMENT: ("a", "b", "c", UNKNOWN),
    s.STATE_HOLIDAY: ("0", "a", "b", "c", UNKNOWN),
}

CALENDAR_FEATURES: tuple[str, ...] = (
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
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

ROW_FEATURES: tuple[str, ...] = (s.PROMO, s.STATE_HOLIDAY, s.SCHOOL_HOLIDAY)

#: The entity's integer id. Never one-hot: real Rossmann has 1,115 stores, which would be
#: 1,115 columns nobody can defend and a distance metric where every pair of stores is
#: equally far apart.
ENTITY_CODE = "entity_code"

_SATURDAY = 5
_MONTHS_PER_YEAR = 12
_DAYS_PER_WEEK = 7


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Everything derivable from the date alone. Knowable arbitrarily far ahead."""
    frame = frame.copy()
    dates = pd.to_datetime(frame[s.DATE])
    frame["day_of_week"] = dates.dt.dayofweek.astype("int16")
    frame["month"] = dates.dt.month.astype("int16")
    frame["day_of_month"] = dates.dt.day.astype("int16")
    frame["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    frame["is_weekend"] = (dates.dt.dayofweek >= _SATURDAY).astype("int8")
    return frame


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


def encode_labels(
    frame: pd.DataFrame, *, encoders: dict[str, LabelEncoder] | None = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode every categorical against its fixed vocabulary.

    Returns the encoders so the artefact can carry them: serving must use exactly the
    mapping training used, and refitting at predict time would silently renumber.
    """
    frame = frame.copy()
    encoders = encoders or {}
    for column, vocabulary in LABEL_VOCABULARIES.items():
        if column not in frame.columns:
            continue
        if column not in encoders:
            encoder = LabelEncoder()
            encoder.fit(list(vocabulary))
            encoders[column] = encoder
        known = set(vocabulary)
        values = frame[column].astype("string").fillna(UNKNOWN)
        values = values.where(values.isin(known), UNKNOWN)
        frame[column] = encoders[column].transform(values.to_numpy()).astype("int16")
    return frame, encoders


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The model matrix's columns, in a stable order."""
    wanted = [ENTITY_CODE, *CALENDAR_FEATURES, *ROW_FEATURES, *STORE_FEATURES]
    return [column for column in wanted if column in frame.columns]


def build(
    sales: pd.DataFrame,
    stores: pd.DataFrame | None,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    age_as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Sales rows plus optional metadata in, model-ready frame out.

    `age_as_of` is a serving concern only — see `add_store_features`. Training leaves it
    unset, so the frame the models are fitted on is unchanged by its existence.
    """
    frame = add_calendar_features(sales)
    frame = add_store_features(frame, stores, age_as_of=age_as_of)
    frame[ENTITY_CODE] = _entity_codes(frame[s.ENTITY])
    frame, encoders = encode_labels(frame, encoders=encoders)

    for column in feature_columns(frame):
        if column not in LABEL_VOCABULARIES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame, encoders


def _entity_codes(entities: pd.Series) -> pd.Series:
    """Stable integer per entity. Numeric ids keep their value; others hash by order."""
    numeric = pd.to_numeric(entities, errors="coerce")
    if not numeric.isna().any():
        return numeric.astype("int32")
    return pd.Series(
        pd.factorize(entities.astype("string"), sort=True)[0], index=entities.index
    ).astype("int32")


def drop_closed(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove days the shop was shut.

    A closed store sells exactly zero and every model predicts zero, so keeping those
    rows adds a large block of free perfect predictions — on Rossmann roughly a seventh
    of them. Every score in this package would be flattered by an amount bought entirely
    by predicting that a shut shop sells nothing.
    """
    if s.OPEN not in frame.columns:
        return frame
    return frame[frame[s.OPEN] == 1].reset_index(drop=True)


def is_constant(values: pd.Series) -> bool:
    """True when a column carries no information, so callers can skip it."""
    return bool(np.asarray(values.nunique(dropna=False)) <= 1)
