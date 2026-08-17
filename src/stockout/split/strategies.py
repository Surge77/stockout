"""The three ways to split this data, side by side, so the difference can be measured.

This is the only module in the package allowed to name a shuffled splitter, and
`tests/test_no_random_splits.py` enforces that by parsing the AST of every other one. The
exemption exists because a project cannot demonstrate what shuffling costs while refusing
to import the thing that shuffles. Showing the mistake and pricing it beats asserting it
is a mistake.

**`random_split`** — `train_test_split(shuffle=True)`, the line in all fourteen of the
course notebooks. On a time series it puts Tuesday in training and the Wednesday either
side of it in test, and since the two share almost every feature value the model is
largely being asked to recall rather than predict.

**`time_holdout`** — everything before a date trains, everything after it tests. What a
forecast actually is.

**`time_series_cv`** — sklearn's `TimeSeriesSplit`, the cross-validated form, used for
hyperparameter search where a single holdout would tune against one accident.

## The trap that makes TimeSeriesSplit lie

`TimeSeriesSplit` slices on **row position**. Every frame in this package arrives sorted
by `KEY_COLUMNS`, which is `(store, date)` — store-major. Handed one of those, it trains
on stores 1..186 and tests on stores 187..372: a badly-implemented `GroupKFold` wearing a
time-series name, producing an entirely plausible R² with nothing raising anywhere.

So every splitter here calls `assert_date_major` first, and `date_major` is the way to
satisfy it. This is not a hypothetical — it is the default state of every frame the rest
of the package produces.

`gap` matters for the same reason `features/lags.py` refuses a lag shorter than the
horizon: a fold whose training rows end the day before its test rows begin is predicting
tomorrow, not the horizon. Pass `gap=horizon` and the two guards agree.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from ..config import DEFAULT_HORIZON_DAYS, DEFAULT_N_FOLDS, RANDOM_SEED
from ..data import schemas as s
from ..errors import BacktestError, LeakageError

#: The share of rows a random split holds out. sklearn's own default, kept so the
#: comparison is against the line a student would actually write.
DEFAULT_TEST_SIZE = 0.25


def date_major(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-sort into `(date, store)` order, which is what a positional splitter needs.

    The package's own `KEY_COLUMNS` order is `(store, date)`, which is right for lag
    construction — `groupby(store).shift()` wants a store's days contiguous — and wrong
    for anything that slices by row position.
    """
    _require_dates(frame)
    keys = [s.DATE] + ([s.STORE] if s.STORE in frame.columns else [])
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def assert_date_major(frame: pd.DataFrame) -> None:
    """Raise unless the frame's dates are non-decreasing down the rows.

    A `LeakageError` rather than a `ValueError`, because that is exactly what a
    positional split of a store-major frame produces, and the error should name the
    thing that goes wrong rather than the type check that caught it.
    """
    _require_dates(frame)
    if not frame[s.DATE].is_monotonic_increasing:
        raise LeakageError(
            "rows are not in date order, so a positional splitter would cut by store "
            "rather than by time. Call split.strategies.date_major first."
        )


def random_split(
    frame: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = RANDOM_SEED,
    stratify_on: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`train_test_split(shuffle=True)`. Correct for tabular data, wrong for this data.

    Kept honest by `stratify_on`, which is what the course notebooks reach for on a
    classification target and which does nothing about the leak — the classes end up
    balanced across two halves that both already know the future.
    """
    stratify = frame[stratify_on] if stratify_on else None
    halves = train_test_split(
        frame, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
    )
    # `train_test_split` is typed as returning a list of anything, because it accepts any
    # number of arrays. One frame in means exactly two frames out.
    train = pd.DataFrame(halves[0])
    test = pd.DataFrame(halves[1])
    return train, test


def time_holdout(
    frame: pd.DataFrame, *, test_days: int = DEFAULT_HORIZON_DAYS, gap_days: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The last `test_days` calendar days test; everything before `gap_days` earlier trains.

    Cut on the calendar rather than on a row count, so every store contributes the same
    dates. A row-count cut would give a store with a refurbishment gap a different test
    window from its neighbours.
    """
    _require_dates(frame)
    if test_days < 1:
        raise ValueError("test_days must be at least 1")
    if gap_days < 0:
        raise ValueError("gap_days cannot be negative")

    last = frame[s.DATE].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    train_end = test_start - pd.Timedelta(days=gap_days + 1)

    train = frame[frame[s.DATE] <= train_end].copy()
    test = frame[frame[s.DATE] >= test_start].copy()
    if train.empty or test.empty:
        raise BacktestError(
            f"a {test_days}-day holdout with a {gap_days}-day gap leaves "
            f"{len(train)} training and {len(test)} test rows"
        )
    return train, test


def time_series_cv(
    frame: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_FOLDS,
    gap: int = DEFAULT_HORIZON_DAYS,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """`TimeSeriesSplit` over a date-major frame, yielding index labels per fold.

    Yields labels rather than positions so a caller can use `.loc` and not silently
    misalign a frame it has filtered since. Refuses a frame that is not date-major,
    because on a store-major one this splits by store — see the module docstring.

    `gap` is in *rows*, which is `TimeSeriesSplit`'s own unit, and with roughly one row
    per store per day it is not the same as days. It is defaulted to the horizon anyway:
    approximately right and pointing the correct direction beats exactly zero.
    """
    assert_date_major(frame)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for train_positions, test_positions in splitter.split(frame):
        yield frame.index[train_positions], frame.index[test_positions]


def _require_dates(frame: pd.DataFrame) -> None:
    if s.DATE not in frame.columns:
        raise BacktestError(f"a split needs a {s.DATE!r} column")
    if frame.empty:
        raise BacktestError("cannot split an empty frame")
