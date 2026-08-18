"""Where the hyperparameters come from, and the three things that make it trustworthy."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data.loaders import merge_store
from stockout.data.synth import make_sales
from stockout.data.synth_stores import make_stores
from stockout.errors import BacktestError, LeakageError
from stockout.features.build import build_features
from stockout.features.store_features import add_store_features
from stockout.models.grids import (
    CLASSIFICATION_GRIDS,
    REGRESSION_GRIDS,
    candidate_count,
    grid_for,
)
from stockout.models.registry import model_names
from stockout.models.tuning import SCORING, tune, tune_many
from stockout.split.strategies import date_major
from stockout.targets import add_demand_class, fit_thresholds

HORIZON = 7


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    built = merge_store(make_sales(n_stores=4, days=420), make_stores(n_stores=4))
    built = build_features(add_store_features(built), horizon=HORIZON)
    built = date_major(built).dropna(subset=["sales_lag_28"])
    return add_demand_class(built, fit_thresholds(built))


# --- the grids are data ---------------------------------------------------------------


def test_every_grid_names_a_registered_model() -> None:
    """A grid for a model that does not exist is a typo nobody would otherwise notice."""
    assert set(REGRESSION_GRIDS) <= set(model_names("regression"))
    assert set(CLASSIFICATION_GRIDS) <= set(model_names("classification"))


def test_every_grid_addresses_the_estimator_step() -> None:
    """`estimate__` is the path through the pipeline the adapter builds. A wrong prefix
    is the classic silent failure of a grid search."""
    for grids in (REGRESSION_GRIDS, CLASSIFICATION_GRIDS):
        for name, grid in grids.items():
            assert all(key.startswith("estimate__") for key in grid), name


def test_every_grid_offers_more_than_one_value_somewhere() -> None:
    for grids in (REGRESSION_GRIDS, CLASSIFICATION_GRIDS):
        for name, grid in grids.items():
            assert candidate_count(grid) > 1, name


def test_a_model_with_nothing_to_tune_gets_an_empty_grid() -> None:
    assert grid_for("linear", task="regression") == {}
    assert candidate_count({}) == 0


def test_the_returned_grid_is_a_copy() -> None:
    """A caller mutating it must not edit the module's own literal."""
    grid = grid_for("ridge", task="regression")
    grid["estimate__alpha"] = [999.0]
    assert grid_for("ridge", task="regression")["estimate__alpha"] != [999.0]


def test_a_grid_brackets_its_range_on_both_sides() -> None:
    """So the chosen value can be read as a choice rather than as a corner."""
    alphas = grid_for("ridge", task="regression")["estimate__alpha"]
    assert min(alphas) < 1.0 < max(alphas)  # type: ignore[operator]


# --- the search ------------------------------------------------------------------------


def test_a_search_finds_a_value_from_its_own_grid(frame: pd.DataFrame) -> None:
    result = tune(frame, "ridge", task="regression", horizon=HORIZON, folds=2)
    assert result.searched
    assert result.best_params["estimate__alpha"] in grid_for("ridge", task="regression")[
        "estimate__alpha"
    ]


def test_a_search_reports_what_it_cost(frame: pd.DataFrame) -> None:
    """Candidates times folds is the number of fits, and it belongs in the report."""
    result = tune(frame, "ridge", task="regression", horizon=HORIZON, folds=2)
    assert result.candidates == 5
    assert result.as_row()["fits"] == 10
    assert result.rows > 0


def test_a_model_with_nothing_to_tune_says_so_rather_than_failing(
    frame: pd.DataFrame,
) -> None:
    result = tune(frame, "linear", task="regression", horizon=HORIZON, folds=2)
    assert not result.searched
    assert result.as_row()["best_params"] == "-"


def test_a_classification_search_optimises_macro_f1_not_accuracy(
    frame: pd.DataFrame,
) -> None:
    """Accuracy would let a search optimise away the smallest class and call it better."""
    assert SCORING["classification"] == "f1_macro"
    result = tune(frame, "logistic", task="classification", horizon=HORIZON, folds=2)
    assert result.searched
    assert -1.0 <= result.best_score <= 1.0


def test_a_store_major_frame_is_refused(frame: pd.DataFrame) -> None:
    """TimeSeriesSplit slices on row position; store-major means it cuts by store."""
    with pytest.raises(LeakageError):
        tune(frame.sort_values(["store", "date"]), "ridge", task="regression", folds=2)


def test_the_search_uses_the_rows_the_model_would_fit_on(frame: pd.DataFrame) -> None:
    """A capped model must be tuned on its cap, or it is tuned for a model nobody builds."""
    capped = tune(frame, "svr", task="regression", horizon=HORIZON, folds=2)
    uncapped = tune(frame, "ridge", task="regression", horizon=HORIZON, folds=2)
    assert capped.rows <= uncapped.rows


def test_tuning_several_models_returns_one_row_each(frame: pd.DataFrame) -> None:
    table = tune_many(frame, ["ridge", "lasso"], task="regression", horizon=HORIZON, folds=2)
    assert list(table["model"]) == ["ridge", "lasso"]


def test_tuning_nothing_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(BacktestError, match="nothing to tune"):
        tune_many(frame, [], task="regression")


def test_the_reported_parameters_drop_the_pipeline_prefix(frame: pd.DataFrame) -> None:
    """`estimate__alpha` is how sklearn addresses a step and is noise to a reader."""
    row = tune(frame, "ridge", task="regression", horizon=HORIZON, folds=2).as_row()
    assert str(row["best_params"]).startswith("alpha=")
