"""The gradient-boosted models.

The parameter choices are asserted so that changing them later is a deliberate diff with
a reason attached, rather than a quiet edit nobody reviews.

LightGBM is fitted once per module here rather than once per test. A quantile forecaster
is five boosters, and refitting it for every assertion buys nothing but wall clock.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.errors import BacktestError, MissingDependencyError
from stockout.models.gbm import (
    DEFAULT_PARAMS,
    DEFAULT_QUANTILES,
    GbmForecaster,
    GbmQuantileForecaster,
)


@pytest.fixture(scope="module")
def point_model(sales: pd.DataFrame) -> GbmForecaster:
    return GbmForecaster(horizon=42).fit(sales)


@pytest.fixture(scope="module")
def quantile_model(sales: pd.DataFrame) -> GbmQuantileForecaster:
    return GbmQuantileForecaster(horizon=42).fit(sales)


def test_the_objective_is_tweedie_because_sales_have_a_point_mass_at_zero() -> None:
    assert DEFAULT_PARAMS["objective"] == "tweedie"


def test_the_default_quantiles_bracket_realistic_critical_ratios() -> None:
    assert 0.5 in DEFAULT_QUANTILES
    assert max(DEFAULT_QUANTILES) >= 0.95
    assert all(0.0 < q < 1.0 for q in DEFAULT_QUANTILES)


def test_quantiles_outside_the_open_unit_interval_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        GbmQuantileForecaster(horizon=42, quantiles=[0.5, 1.0])


def test_a_horizon_below_one_day_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1 day"):
        GbmForecaster(horizon=0)


@pytest.mark.parametrize("build", [GbmForecaster, GbmQuantileForecaster])
def test_boosting_for_no_rounds_is_rejected(build: type) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build(horizon=42, num_boost_round=0)


def test_a_single_quantile_cannot_cross_itself(sales: pd.DataFrame) -> None:
    """One column has no neighbour to cross, so the rate is zero rather than undefined."""
    model = GbmQuantileForecaster(horizon=42, quantiles=[0.5], num_boost_round=20)
    model.fit(sales).predict_quantiles(sales)
    assert model.crossing_rate == 0.0


def test_constructing_a_forecaster_does_not_require_lightgbm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scaffold installs and runs without the `gbm` extra.

    `None` in `sys.modules` is CPython's way of making an import fail, so this asserts
    the absence of LightGBM rather than merely not using it.
    """
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    assert GbmForecaster(horizon=42).horizon == 42
    assert GbmQuantileForecaster(horizon=42).quantiles == DEFAULT_QUANTILES


def test_fitting_without_lightgbm_names_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch, tiny: pd.DataFrame
) -> None:
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(MissingDependencyError, match=r"\[gbm\]"):
        GbmForecaster(horizon=1).fit(tiny)


def test_predicting_before_fitting_is_an_error_rather_than_a_zero(
    sales: pd.DataFrame,
) -> None:
    with pytest.raises(BacktestError, match="before fit"):
        GbmForecaster(horizon=42).predict(sales)
    with pytest.raises(BacktestError, match="before fit"):
        GbmQuantileForecaster(horizon=42).predict_quantiles(sales)


def test_a_training_window_shorter_than_its_lags_fails_loudly(tiny: pd.DataFrame) -> None:
    """Six rows per store cannot support a 42-day lag, and silence would be worse."""
    with pytest.raises(BacktestError, match="shorter than the lags"):
        GbmForecaster(horizon=42).fit(tiny)


def test_customers_never_reaches_the_model(point_model: GbmForecaster) -> None:
    """The denylist is enforced where it matters — in the fitted feature list."""
    assert s.CUSTOMERS not in point_model.features
    assert s.SALES not in point_model.features
    assert point_model.features, "a model with no features would pass vacuously"


def test_the_point_model_beats_seasonal_naive(sales: pd.DataFrame) -> None:
    """The claim the project is trying to earn. It is allowed to fail and be reported."""
    from stockout.evaluate.backtest import backtest

    results = backtest(sales, lambda: GbmForecaster(horizon=42), n_folds=5, horizon=42)
    assert results["mase"].mean() < 1.0


def test_a_closed_store_is_predicted_to_sell_nothing(
    point_model: GbmForecaster, sales: pd.DataFrame
) -> None:
    """Whether a store trades is known in advance, so predicting revenue for a shut one
    is simply wrong rather than optimistic."""
    predicted = point_model.predict(sales)
    assert (predicted[sales[s.OPEN] == 0] == 0.0).all()


def test_quantile_predictions_are_monotonic_across_levels(
    quantile_model: GbmQuantileForecaster, sales: pd.DataFrame
) -> None:
    """Quantile crossing is expected at the tails and must be sorted, not hidden."""
    predicted = quantile_model.predict_quantiles(sales)
    assert (predicted.diff(axis=1).iloc[:, 1:] >= 0).all().all()


def test_the_crossing_rate_is_recorded_rather_than_swallowed(
    quantile_model: GbmQuantileForecaster, sales: pd.DataFrame
) -> None:
    quantile_model.predict_quantiles(sales)
    assert 0.0 <= quantile_model.crossing_rate <= 1.0


def test_the_median_prediction_is_the_half_quantile_column(
    quantile_model: GbmQuantileForecaster, sales: pd.DataFrame
) -> None:
    predicted = quantile_model.predict_quantiles(sales)
    assert quantile_model.predict(sales).equals(predicted["0.5"].rename(None))


def test_the_ninety_percent_quantile_is_roughly_ninety_percent_covered(
    quantile_model: GbmQuantileForecaster, sales: pd.DataFrame
) -> None:
    """A 0.9 quantile covering 99% is not conservative; it is wrong, and it costs money."""
    from stockout.evaluate.metrics import coverage

    predicted = quantile_model.predict_quantiles(sales)
    assert coverage(sales["sales"], predicted["0.9"]) == pytest.approx(0.9, abs=0.05)
