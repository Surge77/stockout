"""Store metadata turned into features that move with the row's date."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.features.store_features import (
    COMPETITION_OPEN_MONTHS,
    IS_PROMO2_MONTH,
    PROMO2_ACTIVE_WEEKS,
    add_store_features,
)


def _frame(**overrides: object) -> pd.DataFrame:
    """One store, four dates spread across the year, with metadata that can be overridden."""
    dates = pd.to_datetime(["2014-01-15", "2014-04-15", "2014-07-15", "2014-10-15"])
    base: dict[str, object] = {
        s.DATE: dates,
        s.STORE: 1,
        s.COMPETITION_OPEN_MONTH: 1.0,
        s.COMPETITION_OPEN_YEAR: 2013.0,
        s.PROMO2_SINCE_WEEK: 1.0,
        s.PROMO2_SINCE_YEAR: 2013.0,
        s.PROMO_INTERVAL: "Jan,Apr,Jul,Oct",
    }
    return pd.DataFrame({**base, **overrides})


# --- competition age -------------------------------------------------------------------


def test_competition_age_counts_whole_months_from_the_opening() -> None:
    """January 2013 to January 2014 is twelve months, and to April 2014 is fifteen."""
    out = add_store_features(_frame())
    assert list(out[COMPETITION_OPEN_MONTHS]) == [12.0, 15.0, 18.0, 21.0]


def test_a_competitor_that_has_not_opened_yet_reads_negative_rather_than_zero() -> None:
    """'Opens in four months' and 'opened four months ago' are different situations."""
    out = add_store_features(_frame(**{s.COMPETITION_OPEN_YEAR: 2015.0}))
    assert (out[COMPETITION_OPEN_MONTHS] < 0).all()


def test_an_unknown_opening_date_stays_unknown() -> None:
    """Imputing here would hide a decision `preprocess.py` makes in the open."""
    out = add_store_features(
        _frame(**{s.COMPETITION_OPEN_MONTH: None, s.COMPETITION_OPEN_YEAR: None})
    )
    assert out[COMPETITION_OPEN_MONTHS].isna().all()


# --- promotion age ---------------------------------------------------------------------


def test_promotion_age_counts_whole_weeks_from_an_iso_week_start() -> None:
    """ISO week 1 of 2013 begins Monday 2013-12-31... of 2012. That is the point.

    `%Y%W` would resolve 2013 week 1 to 2013-01-07 and every duration would be a week
    short. The two conventions disagree exactly where the December and January rows are.
    """
    out = add_store_features(_frame())
    expected_start = pd.Timestamp("2012-12-31")
    expected = (pd.Timestamp("2014-01-15") - expected_start).days / 7
    assert out[PROMO2_ACTIVE_WEEKS].iloc[0] == pytest.approx(expected)


def test_a_store_not_running_the_promotion_has_run_it_for_no_weeks() -> None:
    """Zero, not NaN. The answer is known and it is 'none'."""
    out = add_store_features(_frame(**{s.PROMO2_SINCE_WEEK: None, s.PROMO2_SINCE_YEAR: None}))
    assert (out[PROMO2_ACTIVE_WEEKS] == 0.0).all()


def test_a_promotion_that_starts_later_reads_zero_rather_than_negative() -> None:
    out = add_store_features(_frame(**{s.PROMO2_SINCE_YEAR: 2015.0}))
    assert (out[PROMO2_ACTIVE_WEEKS] == 0.0).all()


# --- the calendar crossing ---------------------------------------------------------------


def test_the_promotion_month_flag_follows_the_rows_own_month() -> None:
    """The whole reason `promo_interval` is worth anything: it stops being a constant."""
    out = add_store_features(_frame())
    assert list(out[IS_PROMO2_MONTH]) == [1, 1, 1, 1]


def test_months_outside_the_interval_are_flagged_off() -> None:
    out = add_store_features(_frame(**{s.PROMO_INTERVAL: "Feb,May,Aug,Nov"}))
    assert list(out[IS_PROMO2_MONTH]) == [0, 0, 0, 0]


def test_rossmanns_four_letter_september_is_understood() -> None:
    """The source file spells it `Sept`. An unmatched token makes the flag silently zero."""
    dates = pd.to_datetime(["2014-09-15"])
    frame = pd.DataFrame({s.DATE: dates, s.PROMO_INTERVAL: ["Mar,Jun,Sept,Dec"]})
    assert add_store_features(frame)[IS_PROMO2_MONTH].iloc[0] == 1


def test_a_store_with_no_promotion_calendar_is_flagged_off_everywhere() -> None:
    out = add_store_features(_frame(**{s.PROMO_INTERVAL: None}))
    assert (out[IS_PROMO2_MONTH] == 0).all()


def test_an_unrecognised_month_token_is_dropped_rather_than_crashing() -> None:
    frame = pd.DataFrame(
        {s.DATE: pd.to_datetime(["2014-04-15"]), s.PROMO_INTERVAL: ["Apr,Smarch"]}
    )
    assert add_store_features(frame)[IS_PROMO2_MONTH].iloc[0] == 1


# --- degrading gracefully ----------------------------------------------------------------


def test_a_frame_without_store_metadata_passes_through_unchanged() -> None:
    """Baseline tests carry no metadata and must not be made to."""
    frame = pd.DataFrame({s.DATE: pd.to_datetime(["2014-01-01"]), s.STORE: [1]})
    assert add_store_features(frame).equals(frame)


def test_a_frame_without_dates_passes_through_unchanged() -> None:
    frame = pd.DataFrame({s.STORE: [1], s.PROMO_INTERVAL: ["Jan,Apr,Jul,Oct"]})
    assert add_store_features(frame).equals(frame)


def test_the_input_frame_is_not_mutated() -> None:
    frame = _frame()
    before = list(frame.columns)
    add_store_features(frame)
    assert list(frame.columns) == before
