"""The gradient-boosted models. Configuration is real; the models are stubs.

The parameter choices are asserted now so that changing them later is a deliberate diff
with a reason attached, rather than a quiet edit nobody reviews.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.models.gbm import (
    DEFAULT_PARAMS,
    DEFAULT_QUANTILES,
    GbmForecaster,
    GbmQuantileForecaster,
)


def test_the_objective_is_tweedie_because_sales_have_a_point_mass_at_zero() -> None:
    assert DEFAULT_PARAMS["objective"] == "tweedie"


def test_the_default_quantiles_bracket_realistic_critical_ratios() -> None:
    assert 0.5 in DEFAULT_QUANTILES
    assert max(DEFAULT_QUANTILES) >= 0.95
    assert all(0.0 < q < 1.0 for q in DEFAULT_QUANTILES)


def test_quantiles_outside_the_open_unit_interval_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="strictly between"):
        GbmQuantileForecaster(horizon=42, quantiles=[0.5, 1.0])


def test_constructing_a_forecaster_does_not_require_lightgbm() -> None:
    """The scaffold installs and tests clean without the `gbm` extra."""
    assert GbmForecaster(horizon=42).horizon == 42
    assert GbmQuantileForecaster(horizon=42).quantiles == DEFAULT_QUANTILES


@pytest.mark.skip(reason="stub — implement in P2, needs `pip install -e '.[gbm]'`")
def test_the_point_model_beats_seasonal_naive(sales: pd.DataFrame) -> None:
    """The claim the project is trying to earn. It is allowed to fail and be reported."""
    from stockout.evaluate.backtest import backtest

    results = backtest(sales, lambda: GbmForecaster(horizon=42), n_folds=5, horizon=42)
    assert results["mase"].mean() < 1.0


@pytest.mark.skip(reason="stub — implement in P3")
def test_quantile_predictions_are_monotonic_across_levels(sales: pd.DataFrame) -> None:
    """Quantile crossing is expected at the tails and must be sorted, not hidden."""
    model = GbmQuantileForecaster(horizon=42).fit(sales)
    predicted = model.predict_quantiles(sales)
    assert (predicted.diff(axis=1).iloc[:, 1:] >= 0).all().all()


@pytest.mark.skip(reason="stub — implement in P3")
def test_the_ninety_percent_quantile_is_roughly_ninety_percent_covered(
    sales: pd.DataFrame,
) -> None:
    """A 0.9 quantile covering 99% is not conservative; it is wrong, and it costs money."""
    from stockout.evaluate.metrics import coverage

    model = GbmQuantileForecaster(horizon=42).fit(sales)
    predicted = model.predict_quantiles(sales)
    assert coverage(sales["sales"], predicted["0.9"]) == pytest.approx(0.9, abs=0.05)
