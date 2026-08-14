"""Calendar features. All future-known, which is why none of them take a horizon."""

from __future__ import annotations

import pandas as pd

from stockout.data import schemas as s
from stockout.features.calendar import (
    CALENDAR_COLUMNS,
    DAYS_SINCE_START,
    IS_STATE_HOLIDAY,
    IS_WEEKEND,
    MONTH,
    YEAR,
    add_calendar,
)


def test_every_declared_calendar_column_is_produced(tiny: pd.DataFrame) -> None:
    out = add_calendar(tiny)
    for column in CALENDAR_COLUMNS:
        assert column in out.columns


def test_the_trend_starts_at_zero(tiny: pd.DataFrame) -> None:
    out = add_calendar(tiny)
    assert out[DAYS_SINCE_START].min() == 0


def test_saturday_and_sunday_are_the_weekend(tiny: pd.DataFrame) -> None:
    out = add_calendar(tiny)
    assert (out.loc[out[s.DAY_OF_WEEK] >= 6, IS_WEEKEND] == 1).all()
    assert (out.loc[out[s.DAY_OF_WEEK] < 6, IS_WEEKEND] == 0).all()


def test_the_holiday_flag_is_set_for_any_non_zero_code(tiny: pd.DataFrame) -> None:
    marked = tiny.copy()
    marked.loc[marked.index[0], s.STATE_HOLIDAY] = "a"
    out = add_calendar(marked)
    assert int(out[IS_STATE_HOLIDAY].iloc[0]) == 1
    assert int(out[IS_STATE_HOLIDAY].iloc[1]) == 0


def test_a_missing_holiday_column_degrades_to_zero(tiny: pd.DataFrame) -> None:
    out = add_calendar(tiny.drop(columns=[s.STATE_HOLIDAY]))
    assert (out[IS_STATE_HOLIDAY] == 0).all()


def test_month_and_year_come_from_the_date(tiny: pd.DataFrame) -> None:
    out = add_calendar(tiny)
    assert out[MONTH].iloc[0] == 1
    assert out[YEAR].iloc[0] == 2024


def test_add_calendar_does_not_mutate_its_input(tiny: pd.DataFrame) -> None:
    before = tiny.copy()
    add_calendar(tiny)
    pd.testing.assert_frame_equal(tiny, before)
