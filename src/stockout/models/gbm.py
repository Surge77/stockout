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

**Why `fit` keeps the training frame.** Every lag here is at least one horizon long, so
the features for a 42-day test window resolve entirely into training history — that is
the whole point of the leakage guard, and it means `predict` cannot build a feature row
from the future frame alone. It needs the history that sits behind it.

LightGBM is imported at call time, not at module import time: `pip install -e .` without
the `gbm` extra must still construct these classes, list them in the CLI, and run the
rest of the package.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..data import schemas as s
from ..errors import BacktestError, MissingDependencyError
from ..features.build import build_features, feature_columns
from .base import open_rows, zero_when_closed

if TYPE_CHECKING:
    import lightgbm as lgb

#: The service-level quantiles worth fitting. 0.5 is the median point forecast; the rest
#: bracket the newsvendor critical ratios that realistic cost pairs produce. 0.75 is the
#: ratio the *default* cost pair produces, and it is on the grid so that the frontier can
#: price the derived target itself rather than the nearest round number to it.
DEFAULT_QUANTILES: tuple[float, ...] = (0.5, 0.75, 0.8, 0.9, 0.95, 0.99)

#: `seed` here is a seeded *sampler* — row and column subsampling inside boosting — and
#: not a seeded *split*. The distinction is the one `tests/test_no_random_splits.py`
#: exists to police: randomness that changes which rows a tree sees is fine, randomness
#: that decides which rows are training and which are testing is not.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
}

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


class _GbmBase:
    """Feature plumbing shared by the point and quantile forecasters."""

    name = "gbm_base"

    def __init__(self, *, horizon: int, params: dict[str, object] | None = None) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1 day")
        self.horizon = horizon
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS if params is None else params)
        self._history = pd.DataFrame()
        self._features: list[str] = []

    @property
    def features(self) -> list[str]:
        """The columns the model was fitted on, in the order it expects them."""
        return list(self._features)

    def _training_matrix(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Design matrix and target, restricted to trading days with defined lags.

        Closed days are dropped for the same reason the baselines drop them: a zero on a
        shut Sunday is not a demand observation, and averaging it in drags every level
        estimate down by roughly a seventh. `zero_when_closed` puts the zeros back at
        prediction time, where they belong.
        """
        self._history = train.copy()
        matrix = build_features(train, horizon=self.horizon)
        usable = open_rows(matrix).dropna(subset=_lag_columns(matrix))
        if usable.empty:
            raise BacktestError(
                f"no training row has a defined lag at horizon {self.horizon}; the "
                "training window is shorter than the lags it implies"
            )
        self._features = feature_columns(usable)
        return usable[self._features], usable[s.SALES]

    def _future_matrix(self, future: pd.DataFrame) -> pd.DataFrame:
        """Feature rows for `future`, built against the retained training history.

        Rows already present in history keep their observed `sales` so that *later*
        rows can lag them. Rows the model has not seen get a NaN target: the protocol
        says a future frame's `sales` column must be treated as absent, and it is only
        there because slicing a frame keeps every column.

        Private, and reached only after a caller has checked that a fit happened.
        """
        keys = pd.MultiIndex.from_frame(future[list(s.KEY_COLUMNS)])
        known = pd.MultiIndex.from_frame(self._history[list(s.KEY_COLUMNS)])

        unseen = future.loc[~np.asarray(keys.isin(known))].reindex(
            columns=self._history.columns
        )
        unseen = unseen.assign(**{s.SALES: np.nan})

        combined = pd.concat([self._history, unseen], ignore_index=True)
        combined = combined.sort_values(list(s.KEY_COLUMNS)).reset_index(drop=True)

        matrix = build_features(combined, horizon=self.horizon)
        # Located rather than reindexed: `store` is itself a feature, so the key columns
        # have to stay in the frame instead of moving into the index. Every key resolves
        # by construction — `combined` is the history plus exactly the rows that were
        # missing from it.
        positions = pd.MultiIndex.from_frame(matrix[list(s.KEY_COLUMNS)]).get_indexer(keys)
        return matrix.iloc[positions][self._features]

    def _predict_with(self, booster: lgb.Booster, future: pd.DataFrame) -> pd.Series:
        raw = np.asarray(booster.predict(self._future_matrix(future)), dtype="float64")
        values = pd.Series(raw, index=future.index).clip(lower=0.0)
        return zero_when_closed(values, future)


class GbmForecaster(_GbmBase):
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


class GbmQuantileForecaster(_GbmBase):
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
        super().__init__(horizon=horizon, params=params)
        if num_boost_round < 1:
            raise ValueError("num_boost_round must be at least 1")
        self.quantiles = tuple(sorted(quantiles))
        self.num_boost_round = num_boost_round
        self.crossing_rate: float = float("nan")
        self._boosters: dict[float, lgb.Booster] = {}

    def fit(self, train: pd.DataFrame) -> GbmQuantileForecaster:
        lightgbm = _lightgbm()
        design, target = self._training_matrix(train)
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


def _lag_columns(matrix: pd.DataFrame) -> list[str]:
    prefix = f"{s.SALES}_lag_"
    return [column for column in matrix.columns if column.startswith(prefix)]


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
