"""Calendar features.

Everything here is knowable arbitrarily far ahead, which is why none of it takes a
horizon argument. That is worth stating explicitly, because the distinction between a
*future-known covariate* and a *lagged observation* is the one people collapse: the
promo calendar for six weeks' time is already decided and sitting in a planning system,
so using it is not leakage. Last Tuesday's sales, relative to a forecast origin six
weeks back, is.
"""

from __future__ import annotations

import pandas as pd

from ..data import schemas as s

#: Derived flag replacing the raw `state_holiday` string, which is categorical text
#: ("0", "a", "b", "c") and cannot go into a numeric matrix as-is.
IS_STATE_HOLIDAY = "is_state_holiday"
IS_WEEKEND = "is_weekend"
MONTH = "month"
DAY_OF_MONTH = "day_of_month"
WEEK_OF_YEAR = "week_of_year"
DAY_OF_YEAR = "day_of_year"
YEAR = "year"
DAYS_SINCE_START = "days_since_start"

CALENDAR_COLUMNS: tuple[str, ...] = (
    MONTH,
    DAY_OF_MONTH,
    WEEK_OF_YEAR,
    DAY_OF_YEAR,
    YEAR,
    DAYS_SINCE_START,
    IS_WEEKEND,
    IS_STATE_HOLIDAY,
)


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Append calendar columns. Pure; does not mutate the input."""
    out = frame.copy()
    dates = out[s.DATE]

    out[MONTH] = dates.dt.month.astype("int8")
    out[DAY_OF_MONTH] = dates.dt.day.astype("int8")
    out[WEEK_OF_YEAR] = dates.dt.isocalendar().week.astype("int8")
    out[DAY_OF_YEAR] = dates.dt.dayofyear.astype("int16")
    out[YEAR] = dates.dt.year.astype("int16")

    # A plain integer trend. GBMs cannot extrapolate it beyond the training range,
    # which is a feature: it lets the model use recent-era level without pretending
    # it can project the trend forward.
    out[DAYS_SINCE_START] = (dates - dates.min()).dt.days.astype("int32")

    out[IS_WEEKEND] = (out[s.DAY_OF_WEEK] >= 6).astype("int8")

    if s.STATE_HOLIDAY in out.columns:
        out[IS_STATE_HOLIDAY] = (out[s.STATE_HOLIDAY].astype(str) != "0").astype("int8")
    else:
        out[IS_STATE_HOLIDAY] = 0

    return out
