"""Three split protocols, and the guard that stops the positional one lying."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.synth import make_sales
from stockout.errors import BacktestError, LeakageError
from stockout.split.strategies import (
    assert_date_major,
    date_major,
    random_split,
    time_holdout,
    time_series_cv,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Four stores, 400 days, in the package's usual store-major order."""
    return make_sales(n_stores=4, days=400)


# --- the guard ------------------------------------------------------------------------


def test_the_packages_own_frames_are_rejected_by_the_guard(frame: pd.DataFrame) -> None:
    """Not a hypothetical. `KEY_COLUMNS` is (store, date), so this is the default state.

    A positional splitter handed one of these cuts by store and calls it time.
    """
    with pytest.raises(LeakageError, match="not in date order"):
        assert_date_major(frame)


def test_date_major_satisfies_the_guard(frame: pd.DataFrame) -> None:
    assert_date_major(date_major(frame))


def test_date_major_keeps_every_row(frame: pd.DataFrame) -> None:
    assert len(date_major(frame)) == len(frame)


def test_date_major_orders_stores_within_a_day(frame: pd.DataFrame) -> None:
    """A stable secondary sort, so two runs produce the same fold membership."""
    ordered = date_major(frame)
    first_day = ordered[ordered[s.DATE] == ordered[s.DATE].min()]
    assert list(first_day[s.STORE]) == sorted(first_day[s.STORE])


def test_a_frame_without_dates_cannot_be_split() -> None:
    with pytest.raises(BacktestError, match="needs a 'date' column"):
        assert_date_major(pd.DataFrame({s.STORE: [1]}))


def test_an_empty_frame_cannot_be_split() -> None:
    with pytest.raises(BacktestError, match="empty"):
        date_major(pd.DataFrame({s.DATE: pd.to_datetime([])}))


# --- the shuffled split, kept so it can be measured ---------------------------------------


def test_a_random_split_holds_out_the_requested_share(frame: pd.DataFrame) -> None:
    train, test = random_split(frame, test_size=0.25)
    assert len(test) == pytest.approx(len(frame) * 0.25, abs=1)
    assert len(train) + len(test) == len(frame)


def test_a_random_split_interleaves_the_two_halves_in_time(frame: pd.DataFrame) -> None:
    """The defect, asserted rather than described: the test set is not the future."""
    train, test = random_split(frame)
    assert test[s.DATE].min() < train[s.DATE].max()
    assert train[s.DATE].min() < test[s.DATE].max()


def test_the_same_seed_produces_the_same_random_split(frame: pd.DataFrame) -> None:
    first, _ = random_split(frame, seed=1)
    second, _ = random_split(frame, seed=1)
    assert first.index.equals(second.index)


def test_a_different_seed_produces_a_different_random_split(frame: pd.DataFrame) -> None:
    first, _ = random_split(frame, seed=1)
    second, _ = random_split(frame, seed=2)
    assert not first.index.equals(second.index)


def test_a_random_split_can_stratify_without_fixing_anything(frame: pd.DataFrame) -> None:
    """Stratifying balances the classes across two halves that both know the future."""
    train, test = random_split(frame, stratify_on=s.DAY_OF_WEEK)
    assert test[s.DATE].min() < train[s.DATE].max()


# --- the time-ordered holdout ----------------------------------------------------------------


def test_a_time_holdout_puts_every_training_day_before_every_test_day(
    frame: pd.DataFrame,
) -> None:
    train, test = time_holdout(frame, test_days=42)
    assert train[s.DATE].max() < test[s.DATE].min()


def test_the_holdout_window_is_the_requested_number_of_days(frame: pd.DataFrame) -> None:
    _, test = time_holdout(frame, test_days=42)
    assert test[s.DATE].nunique() == 42


def test_a_gap_removes_days_from_the_end_of_training_not_from_the_test(
    frame: pd.DataFrame,
) -> None:
    """The horizon-sized gap that stops a fold predicting tomorrow."""
    _, without = time_holdout(frame, test_days=42, gap_days=0)
    train, with_gap = time_holdout(frame, test_days=42, gap_days=7)
    assert with_gap[s.DATE].nunique() == without[s.DATE].nunique()
    assert (with_gap[s.DATE].min() - train[s.DATE].max()).days == 8


def test_every_store_gets_the_same_test_dates(frame: pd.DataFrame) -> None:
    """Cut on the calendar, not on a row count: a store with a gap must not drift."""
    _, test = time_holdout(frame, test_days=20)
    per_store = test.groupby(s.STORE)[s.DATE].agg(["min", "max"])
    assert per_store["min"].nunique() == 1


def test_a_holdout_longer_than_the_data_is_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(BacktestError, match="leaves"):
        time_holdout(frame, test_days=10_000)


@pytest.mark.parametrize("kwargs", [{"test_days": 0}, {"gap_days": -1}])
def test_nonsense_holdout_arguments_are_rejected(
    frame: pd.DataFrame, kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        time_holdout(frame, **kwargs)


# --- cross-validation ----------------------------------------------------------------------


def test_cross_validation_refuses_a_store_major_frame(frame: pd.DataFrame) -> None:
    """The bug this module exists to prevent, asserted at the entry point."""
    with pytest.raises(LeakageError):
        list(time_series_cv(frame))


def test_every_fold_trains_strictly_before_it_tests(frame: pd.DataFrame) -> None:
    ordered = date_major(frame)
    for train_index, test_index in time_series_cv(ordered, n_splits=3, gap=7):
        assert ordered.loc[train_index, s.DATE].max() < ordered.loc[test_index, s.DATE].min()


def test_the_training_window_grows_with_each_fold(frame: pd.DataFrame) -> None:
    ordered = date_major(frame)
    sizes = [len(train) for train, _ in time_series_cv(ordered, n_splits=3, gap=7)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_the_gap_separates_training_from_test(frame: pd.DataFrame) -> None:
    """Without it a fold predicts tomorrow, which `features/lags.py` refuses to build for."""
    ordered = date_major(frame)
    # `gap` counts rows, and this fixture has four stores per day, so 100 rows is about
    # 25 days. Enough to be unambiguous without asking for more folds than fit.
    narrow = next(iter(time_series_cv(ordered, n_splits=3, gap=0)))
    wide = next(iter(time_series_cv(ordered, n_splits=3, gap=100)))

    def separation(fold: tuple[pd.Index, pd.Index]) -> int:
        train_index, test_index = fold
        first_test = ordered.loc[test_index, s.DATE].min()
        last_train = ordered.loc[train_index, s.DATE].max()
        return int((first_test - last_train).days)

    assert separation(wide) > separation(narrow)


def test_folds_are_returned_as_labels_so_a_filtered_frame_cannot_misalign(
    frame: pd.DataFrame,
) -> None:
    ordered = date_major(frame)
    train_index, _ = next(iter(time_series_cv(ordered, n_splits=3)))
    assert train_index.isin(ordered.index).all()
