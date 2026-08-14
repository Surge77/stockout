"""The three forecasters that must be beaten before any model is interesting.

`SeasonalNaive` is the one that matters. Same weekday, last week — no fitting, no
features, no hyperparameters. On retail data it is a genuinely strong forecast, and a
gradient-boosted model that does not clear it has not earned its build time. Every MASE
in this repo is scaled by it, so "did I beat the baseline" is answered by whether a
number is below 1, and cannot be fudged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SEASON_LENGTH_DAYS
from ..data import schemas as s
from .base import open_rows, zero_when_closed


class _StoreLevelBaseline:
    """Shared plumbing: fit a per-store number, fall back outward when a store is new."""

    name = "base"

    def __init__(self) -> None:
        self._by_store: pd.Series = pd.Series(dtype="float64")
        self._global: float = 0.0

    def _fit_fallbacks(self, trading: pd.DataFrame) -> None:
        self._by_store = trading.groupby(s.STORE, observed=True)[s.SALES].mean()
        self._global = float(trading[s.SALES].mean()) if not trading.empty else 0.0

    def _fill(self, values: np.ndarray, future: pd.DataFrame) -> pd.Series:
        store_mean = future[s.STORE].map(self._by_store).to_numpy(dtype="float64")
        values = np.where(np.isnan(values), store_mean, values)
        values = np.where(np.isnan(values), self._global, values)
        return zero_when_closed(pd.Series(values, index=future.index), future)


class NaiveLast(_StoreLevelBaseline):
    """Predict each store's most recent trading-day figure for every future day.

    The floor. It ignores the weekly cycle entirely, which is precisely what makes it
    a useful contrast: the gap between this and `SeasonalNaive` is the value of knowing
    what day of the week it is.
    """

    name = "naive_last"

    def fit(self, train: pd.DataFrame) -> NaiveLast:
        trading = open_rows(train)
        self._last = trading.groupby(s.STORE, observed=True)[s.SALES].last()
        self._fit_fallbacks(trading)
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        values = future[s.STORE].map(self._last).to_numpy(dtype="float64")
        return self._fill(values, future)


class SeasonalNaive(_StoreLevelBaseline):
    """Predict the most recent observation for the same store and same weekday.

    The baseline every other number in this repo is measured against.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = SEASON_LENGTH_DAYS) -> None:
        super().__init__()
        self.season_length = season_length

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        trading = open_rows(train)
        self._by_key = trading.groupby([s.STORE, s.DAY_OF_WEEK], observed=True)[s.SALES].last()
        self._fit_fallbacks(trading)
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([future[s.STORE], future[s.DAY_OF_WEEK]])
        values = self._by_key.reindex(keys).to_numpy(dtype="float64")
        return self._fill(values, future)


class MovingAverage(_StoreLevelBaseline):
    """Mean of each store's last `window` trading days.

    Smoother than `NaiveLast` and, like it, blind to the weekly cycle.
    """

    name = "moving_average"

    def __init__(self, window: int = 28) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least 1 day")
        self.window = window

    def fit(self, train: pd.DataFrame) -> MovingAverage:
        trading = open_rows(train)
        self._mean = trading.groupby(s.STORE, observed=True)[s.SALES].apply(
            lambda series: series.tail(self.window).mean()
        )
        self._fit_fallbacks(trading)
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        values = future[s.STORE].map(self._mean).to_numpy(dtype="float64")
        return self._fill(values, future)


#: Name -> constructor, for the CLI's `--model` flag. Adding a model here is the only
#: step needed to make it runnable from the command line.
BASELINES: dict[str, type] = {
    NaiveLast.name: NaiveLast,
    SeasonalNaive.name: SeasonalNaive,
    MovingAverage.name: MovingAverage,
}
