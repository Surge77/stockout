"""Forecasters: three baselines, and the gradient-boosted point and quantile models.

The baselines import cleanly on a bare install. The gradient-boosted models construct on
one too — LightGBM is imported inside `fit`, so listing a model is never the same thing
as requiring its dependency, and `--model gbm` can appear in `--help` on a machine that
cannot run it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from .baselines import BASELINES
from .gbm import GbmForecaster, GbmQuantileForecaster

#: Models that must be told the horizon they are forecasting at, because their features
#: depend on it. The baselines are not here: same-weekday-last-week is the same rule at
#: every horizon, so there is nothing for them to do with the number.
_HORIZON_AWARE: dict[str, type] = {
    GbmForecaster.name: GbmForecaster,
    GbmQuantileForecaster.name: GbmQuantileForecaster,
}

#: Every name the CLI accepts. Sorted so `--help` does not reorder between runs.
FORECASTER_NAMES: tuple[str, ...] = tuple(sorted({*BASELINES, *_HORIZON_AWARE}))


def forecaster(name: str, *, horizon: int) -> Callable[[], Any]:
    """A zero-argument factory for `name`.

    A factory rather than an instance, because `backtest` builds a fresh model per fold
    and reusing a fitted one leaks the previous fold's fit into the next.
    """
    if name in BASELINES:
        return BASELINES[name]
    if name in _HORIZON_AWARE:
        return partial(_HORIZON_AWARE[name], horizon=horizon)
    raise KeyError(f"unknown forecaster {name!r}; choose from {', '.join(FORECASTER_NAMES)}")
