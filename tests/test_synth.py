"""The generator's four structures, and its determinism."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.synth import make_sales
from stockout.data.validate import calendar_gaps, validate_sales


def test_the_same_seed_produces_an_identical_frame() -> None:
    """`scripts/make_sample.py` depends on this: a regenerated sample must not diff."""
    pd.testing.assert_frame_equal(make_sales(seed=3, days=200), make_sales(seed=3, days=200))


def test_different_seeds_produce_different_data() -> None:
    assert not make_sales(seed=3, days=200).equals(make_sales(seed=4, days=200))


def test_output_satisfies_every_schema_assumption() -> None:
    validate_sales(make_sales(n_stores=4, days=400))


def test_sundays_are_closed_and_sell_nothing() -> None:
    frame = make_sales(n_stores=2, days=120)
    sundays = frame[frame[s.DAY_OF_WEEK] == s.SUNDAY]
    assert not sundays.empty
    assert (sundays[s.OPEN] == 0).all()
    assert (sundays[s.SALES] == 0).all()


def test_one_store_disappears_for_refurbishment() -> None:
    """Absence, not a run of zeros — which is what breaks naive calendar assumptions."""
    frame = make_sales(n_stores=3, days=600)
    gaps = calendar_gaps(frame)
    assert len(gaps) == 1
    assert gaps.iloc[0]["missing_days"] == 90


def test_a_level_shift_is_present_in_the_third_store() -> None:
    frame = make_sales(n_stores=3, days=730)
    store = frame[(frame[s.STORE] == 3) & (frame[s.OPEN] == 1)]
    before = store[store[s.DATE] < store[s.DATE].min() + pd.Timedelta(days=400)][s.SALES].mean()
    after = store[store[s.DATE] > store[s.DATE].min() + pd.Timedelta(days=550)][s.SALES].mean()
    assert after > before * 1.15


def test_the_weekday_cycle_is_larger_than_the_noise() -> None:
    """If it were not, SeasonalNaive and NaiveLast would tie here and differ on Rossmann."""
    frame = make_sales(n_stores=1, days=730)
    trading = frame[frame[s.OPEN] == 1]
    by_weekday = trading.groupby(s.DAY_OF_WEEK)[s.SALES].mean()
    assert by_weekday.max() / by_weekday.min() > 1.3


def test_promotions_lift_sales() -> None:
    frame = make_sales(n_stores=2, days=400)
    trading = frame[frame[s.OPEN] == 1]
    on = trading[trading[s.PROMO] == 1][s.SALES].mean()
    off = trading[trading[s.PROMO] == 0][s.SALES].mean()
    assert on > off


def test_customers_correlate_with_sales() -> None:
    """The whole reason `customers` is on the denylist: it is nearly the answer."""
    trading = make_sales(n_stores=2, days=400)
    trading = trading[trading[s.OPEN] == 1]
    assert trading[s.SALES].corr(trading[s.CUSTOMERS]) > 0.9


@pytest.mark.parametrize("kwargs", [{"n_stores": 0}, {"days": 0}])
def test_rejects_empty_requests(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_sales(**kwargs)
