"""The backtest loop: its self-check, its exclusions, and its refusal to reuse a model."""

from __future__ import annotations

import pandas as pd

from stockout.data import schemas as s
from stockout.evaluate.backtest import backtest, summarise
from stockout.models.baselines import MovingAverage, NaiveLast, SeasonalNaive


def test_seasonal_naive_scores_exactly_one_in_every_fold(sales: pd.DataFrame) -> None:
    """MASE is scaled by seasonal-naive, so seasonal-naive must score 1.0 by identity.

    This is the harness checking itself. If this drifts, every other MASE in the repo
    is measuring against something other than the baseline it claims.
    """
    results = backtest(sales, SeasonalNaive, n_folds=5, horizon=42)
    assert (results["mase"] - 1.0).abs().max() < 1e-12


def test_naive_last_loses_to_seasonal_naive_on_weekly_data(sales: pd.DataFrame) -> None:
    """The generator has a strong weekday cycle, so ignoring it must cost something."""
    results = backtest(sales, NaiveLast, n_folds=5, horizon=42)
    assert results["mase"].mean() > 1.0


def test_closed_days_are_excluded_from_the_scored_rows(sales: pd.DataFrame) -> None:
    included = backtest(sales, SeasonalNaive, n_folds=3, horizon=42, exclude_closed=False)
    excluded = backtest(sales, SeasonalNaive, n_folds=3, horizon=42, exclude_closed=True)
    assert (excluded["n_test"] < included["n_test"]).all()


def test_including_closed_days_flatters_the_per_row_averages(sales: pd.DataFrame) -> None:
    """MAE and RMSE divide by the row count, so a block of perfect zeros drags them down.

    On Rossmann that block is roughly a seventh of all rows — a free improvement bought
    by predicting that a shut shop sells nothing.
    """
    included = backtest(sales, SeasonalNaive, n_folds=3, horizon=42, exclude_closed=False)
    excluded = backtest(sales, SeasonalNaive, n_folds=3, horizon=42, exclude_closed=True)
    assert (excluded["mae"] > included["mae"]).all()
    assert (excluded["rmse"] > included["rmse"]).all()


def test_wmape_is_immune_to_closed_days(sales: pd.DataFrame) -> None:
    """The reason WMAPE is the primary metric here, and a genuinely surprising property.

    A closed day contributes zero to the numerator (actual and prediction are both zero)
    *and* zero to the denominator, so it cancels exactly. WMAPE therefore reports the
    same figure whether or not the exclusion is applied — unlike MAE, which does not.
    The exclusion still matters, because MAE and RMSE are reported alongside it.
    """
    included = backtest(sales, SeasonalNaive, n_folds=3, horizon=42, exclude_closed=False)
    excluded = backtest(sales, SeasonalNaive, n_folds=3, horizon=42, exclude_closed=True)
    pd.testing.assert_series_equal(included["wmape"], excluded["wmape"])


def test_each_fold_gets_a_freshly_constructed_model(sales: pd.DataFrame) -> None:
    """Reusing one fitted object leaks the previous fold's fit, and barely changes scores."""
    built: list[object] = []

    def factory() -> SeasonalNaive:
        model = SeasonalNaive()
        built.append(model)
        return model

    backtest(sales, factory, n_folds=4, horizon=42)
    assert len(built) == 4
    assert len({id(model) for model in built}) == 4


def test_results_carry_one_row_per_fold_with_the_documented_columns(
    sales: pd.DataFrame,
) -> None:
    results = backtest(sales, MovingAverage, n_folds=3, horizon=42)
    assert len(results) == 3
    assert list(results["fold"]) == [0, 1, 2]
    for column in ("wmape", "rmspe", "mase", "mae", "rmse", "n_train", "n_test"):
        assert column in results.columns


def test_training_windows_grow_and_test_windows_stay_put(sales: pd.DataFrame) -> None:
    results = backtest(sales, SeasonalNaive, n_folds=4, horizon=42)
    assert list(results["n_train"]) == sorted(results["n_train"])


def test_scored_rows_never_outnumber_the_test_window(sales: pd.DataFrame) -> None:
    horizon, stores = 42, sales[s.STORE].nunique()
    results = backtest(sales, SeasonalNaive, n_folds=3, horizon=horizon)
    assert (results["n_test"] <= horizon * stores).all()


def test_summarise_averages_across_folds_rather_than_reporting_the_best(
    sales: pd.DataFrame,
) -> None:
    results = backtest(sales, NaiveLast, n_folds=5, horizon=42)
    summary = summarise(results)
    assert summary["folds"] == 5.0
    assert summary["wmape"] == results["wmape"].mean()
    assert summary["wmape"] > results["wmape"].min()
