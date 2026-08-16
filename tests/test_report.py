"""The results table states plainly whether the baseline was beaten."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.evaluate.report import to_markdown


def _results(*mase: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": range(len(mase)),
            "train_start": [pd.Timestamp("2013-01-01")] * len(mase),
            "train_end": [pd.Timestamp("2014-01-01")] * len(mase),
            "test_start": [pd.Timestamp("2014-01-02")] * len(mase),
            "test_end": [pd.Timestamp("2014-02-12")] * len(mase),
            "n_train": [100] * len(mase),
            "n_test": [42] * len(mase),
            "mae": [1000.0] * len(mase),
            "rmse": [1200.0] * len(mase),
            "wmape": [0.15] * len(mase),
            "rmspe": [0.20] * len(mase),
            "mase": list(mase),
        }
    )


def test_an_empty_frame_says_so_rather_than_rendering_a_blank_table() -> None:
    assert "No folds" in to_markdown(pd.DataFrame(), model_name="gbm")


def test_the_table_has_a_header_and_a_separator_rule() -> None:
    lines = to_markdown(_results(0.9), model_name="gbm").splitlines()
    table = [line for line in lines if line.startswith("|")]
    assert table[0].startswith("| fold")
    assert set(table[1]) <= {"|", "-"}
    assert len(table) == 3


def test_beating_the_baseline_is_reported_as_a_percentage() -> None:
    assert "beats seasonal-naive by 10.0%" in to_markdown(_results(0.9), model_name="gbm")


def test_losing_to_the_baseline_is_stated_in_bold_rather_than_softened() -> None:
    out = to_markdown(_results(1.2), model_name="gbm")
    assert "**loses** to seasonal-naive by 20.0%" in out


def test_the_baseline_itself_is_identified_rather_than_claiming_a_tie() -> None:
    assert "this *is* the seasonal-naive baseline" in to_markdown(
        _results(1.0), model_name="seasonal_naive"
    )


def test_an_undefined_mase_is_reported_honestly() -> None:
    assert "MASE undefined" in to_markdown(_results(float("nan")), model_name="gbm")


def test_the_fold_count_appears_in_the_heading() -> None:
    assert "3 rolling-origin folds" in to_markdown(_results(1.0, 1.0, 1.0), model_name="x")


def test_dates_are_rendered_without_a_time_component() -> None:
    out = to_markdown(_results(1.0), model_name="x")
    assert "2013-01-01" in out
    assert "00:00:00" not in out


def test_numbers_are_formatted_to_a_stable_precision() -> None:
    out = to_markdown(_results(0.95), model_name="x")
    assert "0.1500" in out
    assert pytest.approx(0.95) == 0.95


def test_the_frontier_table_names_the_cheapest_service_level() -> None:
    """Leaving the reader to scan for the smallest number is a chart, not an argument."""
    from stockout.evaluate.report import frontier_to_markdown

    table = pd.DataFrame(
        {
            "quantile": [0.5, 0.9],
            "fill_rate": [0.90, 0.99],
            "cycle_service_level": [0.5, 0.9],
            "stockout_days": [12, 2],
            "holding_cost": [100.0, 900.0],
            "shortage_cost": [900.0, 100.0],
            "total_cost": [1000.0, 1000.0 - 1.0],
            "mean_on_hand": [10.0, 90.0],
        }
    )
    out = frontier_to_markdown(table)
    assert "Cheapest at quantile 0.90" in out
    assert "2 short day(s)" in out
    assert "| quantile" in out


def test_an_unpriced_frontier_says_so_rather_than_rendering_an_empty_table() -> None:
    from stockout.evaluate.report import frontier_to_markdown

    assert "No service levels" in frontier_to_markdown(pd.DataFrame())


def test_the_calibration_table_names_the_worst_miss_and_its_direction() -> None:
    """A table of gaps that leaves the reader to find the largest is a chart, not a claim."""
    from stockout.evaluate.report import calibration_to_markdown

    table = pd.DataFrame(
        {
            "quantile": [0.5, 0.9],
            "empirical": [0.52, 0.72],
            "gap": [0.02, -0.18],
            "pinball": [120.0, 80.0],
        }
    )
    out = calibration_to_markdown(table, model_name="gbm_quantile")

    assert "Worst miss at quantile 0.90" in out
    assert "under-covers by 0.180" in out
    assert "72.0% of the days it promises" in out


def test_a_calibration_table_reports_over_covering_as_over_covering() -> None:
    from stockout.evaluate.report import calibration_to_markdown

    table = pd.DataFrame(
        {"quantile": [0.5], "empirical": [0.9], "gap": [0.4], "pinball": [10.0]}
    )
    assert "over-covers by 0.400" in calibration_to_markdown(table, model_name="stub")


def test_an_unscored_model_says_so_rather_than_rendering_an_empty_table() -> None:
    from stockout.evaluate.report import calibration_to_markdown

    assert "No quantiles" in calibration_to_markdown(pd.DataFrame(), model_name="stub")
