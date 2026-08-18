"""Metric arithmetic, including the cases where the honest answer is NaN."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from stockout.evaluate import metrics


def test_wmape_is_total_error_over_total_actual() -> None:
    assert metrics.wmape([100.0, 100.0], [90.0, 130.0]) == pytest.approx(0.2)


def test_wmape_is_nan_when_every_actual_is_zero() -> None:
    """A window of closed days has no defined percentage error. NaN says so; 0 lies."""
    assert np.isnan(metrics.wmape([0.0, 0.0], [5.0, 5.0]))


def test_rmspe_ignores_zero_actuals() -> None:
    """Kaggle's Rossmann metric excludes zero-sales rows, and so does this."""
    with_zero = metrics.rmspe([0.0, 100.0], [50.0, 110.0])
    without_zero = metrics.rmspe([100.0], [110.0])
    assert with_zero == pytest.approx(without_zero)


def test_rmspe_is_nan_when_all_actuals_are_zero() -> None:
    assert np.isnan(metrics.rmspe([0.0, 0.0], [1.0, 1.0]))


def test_mase_is_exactly_one_when_the_model_is_the_baseline() -> None:
    actual, forecast = [10.0, 20.0, 30.0], [12.0, 19.0, 33.0]
    assert metrics.mase(actual, forecast, y_baseline=forecast) == pytest.approx(1.0)


def test_mase_below_one_means_the_baseline_was_beaten() -> None:
    actual = [10.0, 20.0, 30.0]
    better = [10.0, 20.0, 30.0]
    baseline = [12.0, 22.0, 32.0]
    assert metrics.mase(actual, better, y_baseline=baseline) < 1.0


def test_mase_is_nan_when_the_baseline_is_perfect() -> None:
    """Dividing by a zero denominator would report infinite skill for a tie."""
    actual = [10.0, 20.0]
    assert np.isnan(metrics.mase(actual, [11.0, 21.0], y_baseline=actual))


@given(
    st.lists(st.floats(min_value=1.0, max_value=1e6), min_size=1, max_size=40),
    st.floats(min_value=0.1, max_value=10.0),
)
def test_wmape_is_scale_invariant(actual: list[float], factor: float) -> None:
    """Doubling the currency must not change a percentage error."""
    predicted = [value * 1.1 for value in actual]
    scaled_actual = [value * factor for value in actual]
    scaled_predicted = [value * factor for value in predicted]
    assert metrics.wmape(actual, predicted) == pytest.approx(
        metrics.wmape(scaled_actual, scaled_predicted), rel=1e-9
    )


@given(st.lists(st.floats(min_value=0.0, max_value=1e6), min_size=1, max_size=40))
def test_a_perfect_forecast_scores_zero_on_every_error_metric(actual: list[float]) -> None:
    assert metrics.mae(actual, actual) == 0.0
    assert metrics.rmse(actual, actual) == 0.0


@given(st.lists(st.floats(min_value=1.0, max_value=1e6), min_size=1, max_size=40))
def test_a_perfect_forecast_scores_zero_wmape_and_a_perfect_r2(actual: list[float]) -> None:
    """Split from the case above because both metrics need a non-degenerate actual.

    WMAPE divides by the total actual and R2 divides by its variance, so an all-zero
    or constant series makes each of them 0/0 — undefined rather than perfect. The
    strategy starts at 1.0 to keep the total positive; the constant case is covered by
    `test_wmape_is_nan_when_every_actual_is_zero`.
    """
    assert metrics.wmape(actual, actual) == 0.0
    if len(set(actual)) > 1:
        assert metrics.r2(actual, actual) == pytest.approx(1.0)
