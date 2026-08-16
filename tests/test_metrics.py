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


def test_pinball_punishes_under_forecasting_at_a_high_quantile() -> None:
    under = metrics.pinball([100.0], [90.0], tau=0.9)
    over = metrics.pinball([100.0], [110.0], tau=0.9)
    assert under == pytest.approx(9.0)
    assert over == pytest.approx(1.0)


def test_pinball_is_symmetric_at_the_median() -> None:
    under = metrics.pinball([100.0], [90.0], tau=0.5)
    over = metrics.pinball([100.0], [110.0], tau=0.5)
    assert under == pytest.approx(over)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.5])
def test_pinball_rejects_a_tau_outside_the_open_unit_interval(tau: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        metrics.pinball([1.0], [1.0], tau=tau)


def test_coverage_counts_actuals_at_or_below_the_upper_bound() -> None:
    assert metrics.coverage([1.0, 2.0, 3.0, 400.0], [10.0, 10.0, 10.0, 10.0]) == 0.75


def test_empty_input_gives_nan_not_a_crash() -> None:
    assert np.isnan(metrics.mae([], []))
    assert np.isnan(metrics.rmse([], []))
    assert np.isnan(metrics.coverage([], []))
    assert np.isnan(metrics.pinball([], [], tau=0.5))


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
def test_a_perfect_forecast_scores_zero_everywhere(actual: list[float]) -> None:
    assert metrics.mae(actual, actual) == 0.0
    assert metrics.rmse(actual, actual) == 0.0
    assert metrics.pinball(actual, actual, tau=0.7) == 0.0


def test_the_coverage_table_signs_the_gap_so_the_dangerous_direction_reads_negative() -> None:
    """Under-covering sells a promise the shelf does not keep. It must not look like a miss
    in the harmless direction."""
    import pandas as pd

    actual = [10.0] * 10
    predicted = pd.DataFrame({"0.9": [5.0] * 10, "0.5": [50.0] * 10})
    table = metrics.coverage_table(actual, predicted)

    assert table.loc[0, "empirical"] == pytest.approx(0.0)
    assert table.loc[0, "gap"] == pytest.approx(-0.9)
    assert table.loc[1, "gap"] == pytest.approx(0.5)


def test_the_coverage_table_prices_each_level_with_its_own_pinball_loss() -> None:
    """The loss the level was fitted under, so calibration can be judged against accuracy.

    Two columns holding the *same* forecast, so the only thing that can separate their
    losses is the tau each is scored at. A perfect forecast would score zero at every
    level and the assertion would pass even if the column label were ignored entirely —
    which is the bug this test exists to catch.
    """
    import pandas as pd

    # Under-forecast by 10 everywhere. Pinball at tau is then tau * 10.
    predicted = pd.DataFrame({"0.9": [90.0] * 4, "0.5": [90.0] * 4})
    table = metrics.coverage_table([100.0] * 4, predicted)

    assert table.loc[0, "pinball"] == pytest.approx(9.0)
    assert table.loc[1, "pinball"] == pytest.approx(5.0)


def test_a_column_that_is_not_a_quantile_scores_nan_rather_than_crashing_a_report() -> None:
    import pandas as pd

    table = metrics.coverage_table([10.0] * 4, pd.DataFrame({"mean": [10.0] * 4}))
    assert pd.isna(table.loc[0, "quantile"])
    assert pd.isna(table.loc[0, "pinball"])


def test_a_coverage_table_needs_something_to_score() -> None:
    import pandas as pd

    with pytest.raises(ValueError, match="no quantile forecasts"):
        metrics.coverage_table([1.0], pd.DataFrame(index=[0]))
