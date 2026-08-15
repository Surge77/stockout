"""Rolling-origin fold layout, and the invariant that no fold sees its own future."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.errors import BacktestError, LeakageError
from stockout.split.rolling import assert_no_leakage, rolling_origin, split_frame


def test_no_fold_trains_on_the_future(sales: pd.DataFrame) -> None:
    """The whole point. Every training day must strictly precede every test day."""
    for fold in rolling_origin(sales[s.DATE], n_folds=5, horizon=42):
        train, test = split_frame(sales, fold)
        assert train[s.DATE].max() < test[s.DATE].min()


def test_folds_tile_forward_without_overlapping(sales: pd.DataFrame) -> None:
    folds = rolling_origin(sales[s.DATE], n_folds=4, horizon=42)
    assert [f.index for f in folds] == [0, 1, 2, 3]
    for earlier, later in pairwise(folds):
        assert earlier.test_end < later.test_start
        assert later.test_start - earlier.test_end == pd.Timedelta(days=1)


def test_every_test_window_is_exactly_one_horizon(sales: pd.DataFrame) -> None:
    for fold in rolling_origin(sales[s.DATE], n_folds=3, horizon=28):
        assert fold.test_days == 28


def test_last_fold_ends_on_the_last_observation(sales: pd.DataFrame) -> None:
    folds = rolling_origin(sales[s.DATE], n_folds=3, horizon=42)
    assert folds[-1].test_end == sales[s.DATE].max()


def test_gap_leaves_days_unused_between_train_and_test(sales: pd.DataFrame) -> None:
    fold = rolling_origin(sales[s.DATE], n_folds=2, horizon=42, gap=5)[0]
    assert fold.test_start - fold.train_end == pd.Timedelta(days=6)


def test_expanding_windows_all_start_at_the_first_observation(sales: pd.DataFrame) -> None:
    folds = rolling_origin(sales[s.DATE], n_folds=3, horizon=42, expanding=True)
    assert {f.train_start for f in folds} == {sales[s.DATE].min()}


def test_sliding_windows_move_forward(sales: pd.DataFrame) -> None:
    folds = rolling_origin(sales[s.DATE], n_folds=3, horizon=42, expanding=False)
    starts = [f.train_start for f in folds]
    assert starts == sorted(starts)
    assert starts[0] < starts[-1]


def test_too_little_history_raises_rather_than_silently_dropping_folds(
    sales: pd.DataFrame,
) -> None:
    with pytest.raises(BacktestError, match="cannot support"):
        rolling_origin(sales[s.DATE], n_folds=20, horizon=42)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_folds": 0, "horizon": 42}, "n_folds"),
        ({"n_folds": 2, "horizon": 0}, "horizon"),
        ({"n_folds": 2, "horizon": 42, "gap": -1}, "gap"),
    ],
)
def test_rejects_nonsense_arguments(
    sales: pd.DataFrame, kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rolling_origin(sales[s.DATE], **kwargs)


def test_assert_no_leakage_catches_an_overlapping_split(sales: pd.DataFrame) -> None:
    everything = sales
    with pytest.raises(LeakageError, match="shown the future"):
        assert_no_leakage(everything, everything)


def test_assert_no_leakage_rejects_an_empty_side(sales: pd.DataFrame) -> None:
    with pytest.raises(BacktestError, match="empty"):
        assert_no_leakage(sales, sales.iloc[0:0])


def test_fold_reports_its_own_span(sales: pd.DataFrame) -> None:
    fold = rolling_origin(sales[s.DATE], n_folds=2, horizon=42)[0]
    assert fold.train_days == (fold.train_end - fold.train_start).days + 1
