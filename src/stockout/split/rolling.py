"""Rolling-origin splitting, and nothing else.

This module exists to make one mistake impossible rather than discouraged. There is no
`train_test_split` here, no `shuffle` argument, and no random state — a random split of
a time series trains on next month to predict last month, and the resulting score is not
merely optimistic, it is meaningless. `sklearn.model_selection.train_test_split` defaults
to `shuffle=True`, so the wrong thing is also the easy thing; the fix is to not offer it.

Each fold's test window is exactly one horizon long, and the origin walks forward one
horizon per fold, so the folds tile the most recent `n_folds * horizon` days without
overlapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import DEFAULT_MIN_TRAIN_DAYS
from ..data import schemas as s
from ..errors import BacktestError, LeakageError

_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Fold:
    """One train/test division. All bounds are inclusive."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_days(self) -> int:
        return int((self.train_end - self.train_start) / _DAY) + 1

    @property
    def test_days(self) -> int:
        return int((self.test_end - self.test_start) / _DAY) + 1


def rolling_origin(
    dates: pd.Series,
    *,
    n_folds: int,
    horizon: int,
    gap: int = 0,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    expanding: bool = True,
) -> list[Fold]:
    """Lay out `n_folds` folds ending at the last date in `dates`.

    `expanding` keeps every fold's training window anchored at the first observation,
    which is the right default when history is short. Set it False for a sliding window
    of `min_train_days`, which is better when the series has a regime change early on.

    Raises `BacktestError` when the history cannot support the requested layout — an
    explicit failure beats silently returning three folds when five were asked for.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    if gap < 0:
        raise ValueError("gap cannot be negative")

    first, last = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    needed = n_folds * horizon + gap + min_train_days
    available = int((last - first) / _DAY) + 1
    if available < needed:
        raise BacktestError(
            f"{available} days of history cannot support {n_folds} folds at horizon "
            f"{horizon} with a {min_train_days}-day minimum training window "
            f"({needed} days required). Reduce n_folds, horizon, or min_train_days."
        )

    folds: list[Fold] = []
    for index in range(n_folds):
        test_end = last - _DAY * horizon * (n_folds - 1 - index)
        test_start = test_end - _DAY * (horizon - 1)
        train_end = test_start - _DAY * (gap + 1)
        train_start = first if expanding else train_end - _DAY * (min_train_days - 1)
        folds.append(
            Fold(
                index=index,
                train_start=max(train_start, first),
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def split_frame(frame: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice `frame` into (train, test) for one fold, then verify the slice is sane.

    The verification is not paranoia about this function; it is a tripwire for callers
    who build a frame by concatenating sources and hand it over out of order.
    """
    dates = frame[s.DATE]
    train = frame[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test = frame[(dates >= fold.test_start) & (dates <= fold.test_end)]
    assert_no_leakage(train, test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise `LeakageError` unless every training row predates every test row."""
    if train.empty or test.empty:
        raise BacktestError("a fold produced an empty train or test set")

    latest_train = pd.Timestamp(train[s.DATE].max())
    earliest_test = pd.Timestamp(test[s.DATE].min())
    if latest_train >= earliest_test:
        raise LeakageError(
            f"training data runs to {latest_train.date()} but testing starts on "
            f"{earliest_test.date()}; the model would be shown the future"
        )
