"""The date policy: forward is answerable, backward is not, and the two regions differ.

This replaces an earlier test that pinned the opposite rule — everything outside the data
refused. That rule was enforced at the HTTP boundary only, and it confined the panel to
exactly the dates whose true figure is already in the CSV, which is the one region where
`SELECT sales WHERE store = ? AND date = ?` beats every model in the registry.

The policy now splits by direction, and these tests pin the split. A date before the data
still raises: there is nothing behind it. A date after it is answered and marked
`forecast`, because every feature is calendar-shaped or store-shaped and none is a lag, so
nothing about a future day is unknowable. What *is* missing out there is a measured score,
and `DateStatus.caveat` is what carries that fact to a surface.

The guard lives in the library now rather than in `stockout_web`, so the CLI inherits it.
`tests/test_predict_window.py` pins that it actually does.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.errors import StockoutError
from stockout.window import check, parse

START = pd.Timestamp("2013-01-01")
END = pd.Timestamp("2015-12-31")


@pytest.mark.parametrize("date", ["2013-01-01", "2014-06-15", "2015-12-31"])
def test_a_date_inside_the_data_is_observed(date: str) -> None:
    status = check(date, START, END)
    assert status.region == "observed"
    assert status.stamp == pd.Timestamp(date)
    assert not status.is_forecast


def test_the_boundaries_themselves_are_inside() -> None:
    """Inclusive on both ends: the last day with a row is a day that has an actual."""
    assert check("2013-01-01", START, END).region == "observed"
    assert check("2015-12-31", START, END).region == "observed"


@pytest.mark.parametrize("date", ["2016-01-01", "2031-08-27"])
def test_a_date_after_the_data_is_answered_as_a_forecast(date: str) -> None:
    """The change this module exists for. Previously these raised."""
    status = check(date, START, END)
    assert status.region == "forecast"
    assert status.is_forecast


@pytest.mark.parametrize("date", ["2012-12-31", "1999-01-01"])
def test_a_date_before_the_data_still_raises(date: str) -> None:
    with pytest.raises(StockoutError, match="before the data"):
        check(date, START, END)


def test_the_refusal_names_the_date_it_would_accept() -> None:
    """The message reaches the user, so it has to say what would work instead."""
    with pytest.raises(StockoutError) as caught:
        check("2010-05-01", START, END)
    assert "2013-01-01" in str(caught.value)


def test_days_beyond_counts_from_the_end_of_the_data() -> None:
    assert check("2016-01-01", START, END).days_beyond == 1
    assert check("2016-12-30", START, END).days_beyond == 365


def test_days_beyond_is_zero_inside() -> None:
    assert check("2014-06-15", START, END).days_beyond == 0


def test_a_forecast_carries_a_caveat_and_an_observed_date_does_not() -> None:
    """A surface that words both regions the same way is the failure this prevents."""
    assert check("2014-06-15", START, END).caveat() is None

    caveat = check("2016-06-15", START, END).caveat()
    assert caveat is not None
    assert "2015-12-31" in caveat
    assert "no measured score" in caveat.lower()


@pytest.mark.parametrize("date", ["", "not a date", "2015-13-45"])
def test_an_unparseable_string_raises_stockout_error(date: str) -> None:
    """Not a bare `ValueError`. The router catches `StockoutError` and re-renders the
    form with the message; anything else escapes as a 500."""
    with pytest.raises(StockoutError):
        parse(date)
    with pytest.raises(StockoutError):
        check(date, START, END)
