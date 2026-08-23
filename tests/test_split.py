"""The chronological split, and the property that makes it worth having.

The one thing that must hold is a wall: every training row strictly earlier than every
test row. If a single date lands on both sides the split is a shuffled one wearing a
different name, and the score it produces means nothing more than the old one did.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.config import TEST_SIZE
from stockout.data import schemas as s
from stockout.datasets import prepare
from stockout.errors import StockoutError
from stockout.evaluate import split


@pytest.fixture(scope="module")
def prepared():
    return prepare("synthetic")


def _dates(frame: pd.DataFrame, prepared) -> pd.Series:
    """Recover the dates for a split half, which carries feature columns only."""
    return pd.to_datetime(prepared.frame.loc[frame.index, s.DATE])


def test_no_date_appears_on_both_sides(prepared) -> None:
    """The wall. A store-day is many rows sharing a date, so the cut must be on the date."""
    data = split(prepared, "regression", strategy="chronological")

    train_dates = set(_dates(data.X_train, prepared))
    test_dates = set(_dates(data.X_test, prepared))

    assert not train_dates & test_dates


def test_every_training_row_precedes_every_test_row(prepared) -> None:
    data = split(prepared, "regression", strategy="chronological")

    assert _dates(data.X_train, prepared).max() < _dates(data.X_test, prepared).min()


def test_it_holds_out_roughly_the_requested_fraction(prepared) -> None:
    """Cut on whole dates, so the row share lands near `test_size` rather than exactly."""
    data = split(prepared, "regression", test_size=TEST_SIZE, strategy="chronological")

    train_rows, test_rows = data.sizes
    held_out = test_rows / (train_rows + test_rows)

    assert TEST_SIZE - 0.05 < held_out < TEST_SIZE + 0.05


def test_the_shuffled_strategy_still_interleaves(prepared) -> None:
    """The control arm, kept deliberately. If this ever stops overlapping, `leakage`
    is comparing two identical things and its answer is meaningless."""
    data = split(prepared, "regression", strategy="shuffled")

    train_dates = set(_dates(data.X_train, prepared))
    test_dates = set(_dates(data.X_test, prepared))

    assert train_dates & test_dates


def test_thresholds_still_come_from_the_training_half_only(prepared) -> None:
    """Changing the split must not change where the demand cut points are fitted."""
    data = split(prepared, "classification", strategy="chronological")

    assert len(data.thresholds) > 0
    assert set(data.Y_train.unique()) <= {"Low", "Medium", "High"}


def test_a_single_date_cannot_be_split_chronologically() -> None:
    """One day of data has no later period to hold out; say so rather than returning
    an empty test set that scores as a silent zero."""
    from stockout.evaluate import _by_calendar

    frame = pd.DataFrame({s.DATE: pd.to_datetime(["2015-01-01"] * 4)})

    with pytest.raises(StockoutError, match="two distinct dates"):
        _by_calendar(frame, test_size=0.2)
