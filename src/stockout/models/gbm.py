"""Gradient-boosted forecasters — point and quantile.

**Why LightGBM and not a neural network.** On tabular retail with strong calendar
structure, gradient-boosted trees win, train in seconds, and can be explained in an
interview. A sequence model adds days of build time, a GPU dependency and a story that
ends in "and it was slightly worse". Recorded in ADR 0002.

**Why the tweedie objective.** Sales are non-negative, continuous above zero, and have a
point mass at zero (closures, dead days). That is a compound Poisson-gamma, which is what
`objective="tweedie"` fits. Squared error on the raw target over-weights the high tail;
squared error on `log1p` biases the back-transformed mean low. Try tweedie first, and
report the comparison rather than asserting it.

**Why quantiles rather than a point forecast plus a normal safety stock.** The usual
`mu + z * sigma` rule assumes symmetric, constant-variance errors. Retail demand errors
are neither: variance scales with level and the right tail is longer. Fitting the target
quantile directly makes no distributional assumption, and its calibration is measurable.
Recorded in ADR 0007.

The design matrix, the retained history and the leakage guards live in `design.py`. What
is here is the objective, the number of rounds, and what to do when six boosters disagree
about a spread.

LightGBM is imported at call time, not at module import time: `pip install -e .` without
the `gbm` extra must still construct these classes, list them in the CLI, and run the
rest of the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..errors import BacktestError, MissingDependencyError
from .base import zero_when_closed
from .design import DEFAULT_PARAMS, GbmDesign

if TYPE_CHECKING:
    import lightgbm as lgb

__all__ = [
    "DEFAULT_NUM_BOOST_ROUND",
    "DEFAULT_PARAMS",
    "DEFAULT_QUANTILES",
    "GbmForecaster",
    "GbmQuantileForecaster",
]

#: The service-level quantiles worth fitting. 0.5 is the median point forecast; the rest
#: bracket the newsvendor critical ratios that realistic cost pairs produce. 0.75 is the
#: ratio the *default* cost pair produces, and it is on the grid so that the frontier can
#: price the derived target itself rather than the nearest round number to it.
DEFAULT_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.8, 0.9, 0.95, 0.99)

DEFAULT_NUM_BOOST_ROUND = 800


def _lightgbm() -> Any:
    """Import LightGBM on demand, and say what to install when it is absent."""
    try:
        import lightgbm
    except ImportError as exc:
        raise MissingDependencyError(
            "LightGBM is not installed. The baselines and the backtest harness run "
            "without it; the gradient-boosted models do not. Install with "
            "`pip install -e \".[gbm]\"`."
        ) from exc
    return lightgbm


class GbmForecaster(GbmDesign):
    """LightGBM point forecaster over the horizon-aware feature matrix."""

    name = "gbm"

    def __init__(
        self,
        *,
        horizon: int,
        params: dict[str, object] | None = None,
        num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    ) -> None:
        super().__init__(horizon=horizon, params=params)
        if num_boost_round < 1:
            raise ValueError("num_boost_round must be at least 1")
        self.num_boost_round = num_boost_round
        self._booster: lgb.Booster | None = None

    def fit(self, train: pd.DataFrame) -> GbmForecaster:
        """Build features, drop rows whose lags are undefined, and boost."""
        lightgbm = _lightgbm()
        design, target = self._training_matrix(train)
        dataset = lightgbm.Dataset(design, label=target, free_raw_data=False)
        self._booster = lightgbm.train(
            self.params, dataset, num_boost_round=self.num_boost_round
        )
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        if self._booster is None:
            raise BacktestError("predict() was called before fit()")
        return self._predict_with(self._booster, future)


class GbmQuantileForecaster(GbmDesign):
    """One LightGBM model per quantile, sharing a feature matrix.

    Quantile crossing (the 0.8 prediction landing above the 0.9) is expected at the
    tails. Each row's quantile vector is sorted, and the share of rows that needed
    sorting is kept on `crossing_rate` rather than hidden — a model that crosses on a
    third of its rows is telling you something about the tails.
    """

    name = "gbm_quantile"

    def __init__(
        self,
        *,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        params: dict[str, object] | None = None,
        num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    ) -> None:
        if any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        if not quantiles:
            raise ValueError("at least one quantile is required")
        # Boosters are stored per quantile, so a repeated level would silently drop a
        # column the caller asked for — and a missing column in a frontier is a missing
        # option, not a cosmetic difference.
        if len(set(quantiles)) != len(quantiles):
            raise ValueError("quantiles must be distinct")
        super().__init__(horizon=horizon, params=params)
        if num_boost_round < 1:
            raise ValueError("num_boost_round must be at least 1")
        self.quantiles = tuple(sorted(quantiles))
        self.num_boost_round = num_boost_round
        self.crossing_rate: float = float("nan")
        self._boosters: dict[float, lgb.Booster] = {}

    def fit(self, train: pd.DataFrame) -> GbmQuantileForecaster:
        return self._fit(train, history=None)

    def fit_within(
        self, train: pd.DataFrame, *, history: pd.DataFrame
    ) -> GbmQuantileForecaster:
        """Boost on `train`'s rows while lagging against the whole of `history`.

        The one caller is conformal calibration in its no-refit mode: a model that has
        never trained on the calibration window still has to build features across it,
        because a test row's lag lands inside those days. ADR 0013.
        """
        return self._fit(train, history=history)

    def _fit(
        self, train: pd.DataFrame, *, history: pd.DataFrame | None
    ) -> GbmQuantileForecaster:
        lightgbm = _lightgbm()
        design, target = self._training_matrix(train, history=history)
        self._boosters = {}
        for quantile in self.quantiles:
            # A fresh Dataset per quantile: LightGBM binds construction parameters to a
            # Dataset on first use and refuses to retrain it under a different objective.
            dataset = lightgbm.Dataset(design, label=target, free_raw_data=False)
            self._boosters[quantile] = lightgbm.train(
                _quantile_params(self.params, quantile),
                dataset,
                num_boost_round=self.num_boost_round,
            )
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """The median. Use `predict_quantiles` for the full set."""
        predicted = self.predict_quantiles(future)
        median = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return predicted[_label(median)].rename(None)

    def predict_quantiles(self, future: pd.DataFrame) -> pd.DataFrame:
        """One column per fitted quantile, monotonically sorted per row."""
        if not self._boosters:
            raise BacktestError("predict_quantiles() was called before fit()")

        # Built once and shared across the six boosters. Calling `_predict_with` per
        # quantile would rebuild the same matrix six times, and building it means
        # re-deriving every lag over the whole retained history.
        design = self._future_matrix(future)
        columns = {
            _label(quantile): zero_when_closed(
                pd.Series(
                    np.asarray(booster.predict(design), dtype="float64"),
                    index=future.index,
                ).clip(lower=0.0),
                future,
            )
            for quantile, booster in self._boosters.items()
        }
        raw = pd.DataFrame(columns, index=future.index)
        self.crossing_rate = _crossing_rate(raw)
        return pd.DataFrame(
            np.sort(raw.to_numpy(dtype="float64"), axis=1),
            index=raw.index,
            columns=raw.columns,
        )


def _label(quantile: float) -> str:
    """Column name for a quantile. `0.9` is a better column header than `q90`."""
    return str(quantile)


def _quantile_params(base: dict[str, Any], quantile: float) -> dict[str, Any]:
    """Swap the tweedie objective for a pinball loss at `quantile`."""
    params = {k: v for k, v in base.items() if not k.startswith("tweedie")}
    params["objective"] = "quantile"
    params["alpha"] = quantile
    return params


def _crossing_rate(predicted: pd.DataFrame) -> float:
    """Share of rows whose quantile predictions were not already in order."""
    if predicted.empty or predicted.shape[1] < 2:
        return 0.0
    crossed = (predicted.diff(axis=1).iloc[:, 1:] < 0).any(axis=1)
    return float(crossed.mean())
