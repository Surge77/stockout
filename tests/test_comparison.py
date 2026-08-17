"""The table every notebook, the admin page and results.md read from."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.dataset import prepare
from stockout.evaluate.comparison import (
    CLASSIFICATION_COLUMNS,
    REGRESSION_COLUMNS,
    compare,
    to_markdown,
)
from stockout.targets import DEMAND_CLASS, fit_thresholds

TEST_DAYS = 21
GAP = 7


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return prepare(horizon=7).frame


@pytest.fixture(scope="module")
def regression(frame: pd.DataFrame) -> pd.DataFrame:
    return compare(
        frame, task="regression", test_days=TEST_DAYS, gap_days=GAP, models=["dummy", "ridge"]
    )


@pytest.fixture(scope="module")
def classification(frame: pd.DataFrame) -> pd.DataFrame:
    return compare(
        frame,
        task="classification",
        test_days=TEST_DAYS,
        gap_days=GAP,
        models=["dummy", "logistic"],
    )


def test_the_regression_table_has_its_columns(regression: pd.DataFrame) -> None:
    assert list(regression.columns) == list(REGRESSION_COLUMNS)


def test_the_classification_table_has_its_columns(classification: pd.DataFrame) -> None:
    assert list(classification.columns) == list(CLASSIFICATION_COLUMNS)


def test_models_appear_in_the_order_asked_for(regression: pd.DataFrame) -> None:
    """Registry order answers 'did the extra complexity pay', which sorting by score cannot."""
    assert list(regression["model"]) == ["dummy", "ridge"]


def test_training_and_test_rows_are_separate_columns(regression: pd.DataFrame) -> None:
    """They collided under one key once, and the scoring half won — which erased the fact
    that the capped models see far fewer training rows than the others. ADR 0015."""
    assert (regression["train_rows"] > regression["test_rows"]).all()


def test_wall_clock_is_reported(regression: pd.DataFrame) -> None:
    """A table with only accuracy quietly recommends the forty-minute model."""
    assert (regression["seconds"] >= 0).all()


def test_every_row_carries_the_reason_the_model_is_in_the_comparison(
    regression: pd.DataFrame,
) -> None:
    assert regression["note"].str.len().gt(0).all()


def test_a_real_model_beats_the_dummy_on_both_halves(
    regression: pd.DataFrame, classification: pd.DataFrame
) -> None:
    """If it does not, the floor is doing the work and the table is measuring nothing."""
    def cell(table: pd.DataFrame, model: str, column: str) -> float:
        """`.to_dict()` because a type checker cannot narrow `.at[...]` to a scalar."""
        rows = {str(row["model"]): row for row in table.to_dict(orient="records")}
        return float(rows[model][column])

    assert cell(regression, "ridge", "r2") > 0.5
    assert cell(classification, "logistic", "macro_f1") > cell(
        classification, "dummy", "macro_f1"
    )


def test_the_thresholds_are_refitted_on_the_training_slice(frame: pd.DataFrame) -> None:
    """`prepare` fits them over everything, which is leakage once a model is scored."""
    from stockout.split.strategies import time_holdout

    train, _ = time_holdout(frame, test_days=TEST_DAYS, gap_days=GAP)
    assert fit_thresholds(train).for_store(1) != fit_thresholds(frame).for_store(1)
    assert DEMAND_CLASS in frame.columns


def test_the_markdown_names_the_winner_on_the_deciding_metric(
    regression: pd.DataFrame,
) -> None:
    """Lower is better for WMAPE and higher for macro-F1; hard-coding either names a loser."""
    assert "wins on wmape" in to_markdown(regression, task="regression")


def test_the_classification_markdown_decides_on_macro_f1(
    classification: pd.DataFrame,
) -> None:
    assert "wins on macro_f1" in to_markdown(classification, task="classification")


def test_an_empty_table_says_so_rather_than_rendering_a_header() -> None:
    assert "Nothing was compared" in to_markdown(pd.DataFrame(), task="regression")
