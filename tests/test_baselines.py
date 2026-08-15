"""The three baselines, and the trading-calendar rule they all obey."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.models.base import Forecaster, open_rows, zero_when_closed
from stockout.models.baselines import BASELINES, MovingAverage, NaiveLast, SeasonalNaive


def _future(store: int, day_of_week: int, *, is_open: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            s.STORE: [store],
            s.DAY_OF_WEEK: [day_of_week],
            s.OPEN: [is_open],
            s.DATE: [pd.Timestamp("2024-06-01")],
        }
    )


def test_seasonal_naive_returns_the_last_same_weekday_value(tiny: pd.DataFrame) -> None:
    model = SeasonalNaive().fit(tiny)
    store_one = tiny[tiny[s.STORE] == 1]
    latest_monday = store_one[store_one[s.DAY_OF_WEEK] == 1].iloc[-1]

    predicted = model.predict(_future(1, 1))
    assert predicted.iloc[0] == pytest.approx(latest_monday[s.SALES])


def test_naive_last_ignores_the_weekday(tiny: pd.DataFrame) -> None:
    model = NaiveLast().fit(tiny)
    last_value = tiny[tiny[s.STORE] == 1][s.SALES].iloc[-1]
    for day_of_week in (1, 3, 5):
        assert model.predict(_future(1, day_of_week)).iloc[0] == pytest.approx(last_value)


def test_moving_average_uses_only_the_trailing_window(tiny: pd.DataFrame) -> None:
    """Store 1 runs 100..105, so the last two days average 104.5."""
    model = MovingAverage(window=2).fit(tiny)
    assert model.predict(_future(1, 2)).iloc[0] == pytest.approx(104.5)


def test_moving_average_rejects_a_zero_window() -> None:
    with pytest.raises(ValueError, match="at least 1 day"):
        MovingAverage(window=0)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_predicts_zero_for_a_closed_store(name: str, tiny: pd.DataFrame) -> None:
    model = BASELINES[name]().fit(tiny)
    assert model.predict(_future(1, 2, is_open=0)).iloc[0] == 0.0


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_satisfies_the_forecaster_protocol(name: str) -> None:
    assert isinstance(BASELINES[name](), Forecaster)


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_an_unseen_store_falls_back_instead_of_producing_nan(
    name: str, tiny: pd.DataFrame
) -> None:
    """A store opened after the training window must still get a number."""
    model = BASELINES[name]().fit(tiny)
    predicted = model.predict(_future(999, 3))
    assert predicted.notna().all()
    assert predicted.iloc[0] > 0


def test_baselines_fit_on_trading_days_only(tiny: pd.DataFrame) -> None:
    """A closed day's zero is not a demand observation; averaging it in drags the level down."""
    closure = tiny.tail(1).assign(
        **{s.OPEN: 0, s.SALES: 0.0, s.DATE: pd.Timestamp("2024-01-07")}
    )
    with_closure = pd.concat([tiny, closure], ignore_index=True)
    assert MovingAverage(window=10).fit(with_closure).predict(_future(2, 1)).iloc[0] > 0


def test_open_rows_is_a_no_op_when_the_column_is_absent(tiny: pd.DataFrame) -> None:
    without_open = tiny.drop(columns=[s.OPEN])
    assert len(open_rows(without_open)) == len(without_open)


def test_zero_when_closed_is_a_no_op_when_the_column_is_absent(tiny: pd.DataFrame) -> None:
    predictions = pd.Series([5.0] * len(tiny), index=tiny.index)
    without_open = tiny.drop(columns=[s.OPEN])
    pd.testing.assert_series_equal(zero_when_closed(predictions, without_open), predictions)
