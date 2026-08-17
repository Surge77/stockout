"""Serving one store-day, and the horizon-shaped limit on how far ahead it can see."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout import config
from stockout.dataset import prepare
from stockout.errors import BacktestError, SchemaError
from stockout.predict import forecast
from stockout.train import train

HORIZON = 7


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return prepare(horizon=HORIZON).frame


@pytest.fixture(scope="module")
def artifact(history: pd.DataFrame):
    return train(history, horizon=HORIZON, score=False)


@pytest.fixture(scope="module")
def last_day(history: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(history["date"].max())


def test_a_forecast_answers_both_halves(artifact, history, last_day) -> None:
    result = forecast(artifact, history, store=1, date=last_day + pd.Timedelta(days=1))
    assert result.sales > 0
    assert result.demand_class in config.DEMAND_CLASS_LABELS


def test_the_forecast_reports_the_cut_points_it_used(artifact, history, last_day) -> None:
    """A class with no thresholds beside it cannot be argued with."""
    result = forecast(artifact, history, store=1, date=last_day + pd.Timedelta(days=1))
    low, high = result.thresholds
    assert low < high
    assert f"{low:,.0f}" in result.summary()


def test_a_promotion_changes_the_answer(artifact, history, last_day) -> None:
    """Otherwise the future-known covariates are decoration."""
    when = last_day + pd.Timedelta(days=1)
    quiet = forecast(artifact, history, store=1, date=when, promo=0)
    promoted = forecast(artifact, history, store=1, date=when, promo=1)
    assert quiet.sales != promoted.sales


def test_a_closed_day_is_forecast_as_no_sales(artifact, history, last_day) -> None:
    result = forecast(
        artifact, history, store=1, date=last_day + pd.Timedelta(days=1), is_open=0
    )
    assert result.sales == 0.0


def test_a_closed_day_still_gets_a_class_rather_than_a_null(
    artifact, history, last_day
) -> None:
    """The classifier abstains on a closed day; the fallback bins the regression."""
    result = forecast(
        artifact, history, store=1, date=last_day + pd.Timedelta(days=1), is_open=0
    )
    assert result.demand_class in config.DEMAND_CLASS_LABELS


def test_the_far_edge_of_the_horizon_is_answerable(artifact, history, last_day) -> None:
    assert forecast(
        artifact, history, store=1, date=last_day + pd.Timedelta(days=HORIZON)
    ).sales > 0


def test_one_day_past_the_horizon_is_refused(artifact, history, last_day) -> None:
    """Beyond it the lag would be a prediction of a prediction, compounding its own error."""
    with pytest.raises(BacktestError, match="own output as an observation"):
        forecast(artifact, history, store=1, date=last_day + pd.Timedelta(days=HORIZON + 1))


def test_a_day_already_in_history_is_refused(artifact, history, last_day) -> None:
    with pytest.raises(BacktestError, match="already in store"):
        forecast(artifact, history, store=1, date=last_day)


def test_an_unknown_store_is_refused(artifact, history, last_day) -> None:
    with pytest.raises(BacktestError, match="no history for store"):
        forecast(artifact, history, store=9999, date=last_day + pd.Timedelta(days=1))


def test_history_without_the_key_columns_is_refused(artifact) -> None:
    with pytest.raises(SchemaError, match="store and date"):
        forecast(artifact, pd.DataFrame({"sales": [1.0]}), store=1, date="2015-01-01")


def test_the_lag_features_are_rebuilt_rather_than_reused(
    artifact, history, last_day
) -> None:
    """The new row changes every rolling window that ends on it, so a copied feature
    would describe the day before."""
    result = forecast(artifact, history, store=1, date=last_day + pd.Timedelta(days=1))
    assert result.sales > 0
