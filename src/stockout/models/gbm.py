"""Gradient-boosted forecasters — point and quantile. STUB.

Nothing here is implemented. The signatures, the chosen objective and the reasoning are
committed first so that the decision is on record before the result is known, and so the
backtest harness can be pointed at a real model without changing a line of the harness.

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

Install with `pip install -e ".[gbm]"` before implementing.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: The service-level quantiles worth fitting. 0.5 is the median point forecast; the rest
#: bracket the newsvendor critical ratios that realistic cost pairs produce.
DEFAULT_QUANTILES: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.99)

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
}


class GbmForecaster:
    """LightGBM point forecaster over the horizon-aware feature matrix. STUB."""

    name = "gbm"

    def __init__(
        self,
        *,
        horizon: int,
        params: dict[str, object] | None = None,
        num_boost_round: int = 800,
    ) -> None:
        self.horizon = horizon
        self.params = dict(DEFAULT_PARAMS if params is None else params)
        self.num_boost_round = num_boost_round

    def fit(self, train: pd.DataFrame) -> GbmForecaster:
        """Build features, drop rows whose lags are undefined, and boost."""
        raise NotImplementedError("see docs/decisions/0002-gradient-boosting-not-deep-learning.md")

    def predict(self, future: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class GbmQuantileForecaster:
    """One LightGBM model per quantile, sharing a feature matrix. STUB.

    Quantile crossing (the 0.8 prediction landing above the 0.9) is expected at the
    tails and must be handled — sort the per-row quantile vector, and report how often
    it happened rather than hiding it.
    """

    name = "gbm_quantile"

    def __init__(
        self,
        *,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        params: dict[str, object] | None = None,
    ) -> None:
        if any(not 0.0 < q < 1.0 for q in quantiles):
            raise ValueError("quantiles must lie strictly between 0 and 1")
        self.horizon = horizon
        self.quantiles = tuple(quantiles)
        self.params = dict(DEFAULT_PARAMS if params is None else params)

    def fit(self, train: pd.DataFrame) -> GbmQuantileForecaster:
        raise NotImplementedError

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """The median. Use `predict_quantiles` for the full set."""
        raise NotImplementedError

    def predict_quantiles(self, future: pd.DataFrame) -> pd.DataFrame:
        """One column per fitted quantile, monotonically sorted per row."""
        raise NotImplementedError
