"""Forecast accuracy metrics, and one metric deliberately absent.

**MAPE is not here.** It divides by the actual, and Rossmann is full of zeros and
near-zeros, so it either explodes or has to be patched with an epsilon that quietly
decides the answer. It also punishes over-forecasting more than under-forecasting, which
for a stock decision is exactly backwards. WMAPE divides by the *total* actual instead
and has neither problem. Recorded in ADR 0005.

**MASE here is the same-window form.** Classical MASE scales by the in-sample naive MAE.
This scales by a supplied baseline's MAE over the *evaluation* window, because the claim
that matters is "did this beat seasonal-naive on the data it was tested on", and the
number should say literally that: below 1 means beaten, 1.0 means tied.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as sk_r2_score

#: Anything `np.asarray` can turn into a float vector. Plain lists are included on
#: purpose: a metric test reads better with literal numbers than with constructors.
ArrayLike: TypeAlias = "np.ndarray | pd.Series | Sequence[float]"


def _as_array(values: ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    actual, predicted = _as_array(y_true), _as_array(y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    actual, predicted = _as_array(y_true), _as_array(y_pred)
    if actual.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wmape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted mean absolute percentage error: sum|error| / sum|actual|.

    Returns NaN when the actuals sum to zero, rather than infinity — a window of
    closed days has no defined percentage error, and NaN propagates honestly.
    """
    actual, predicted = _as_array(y_true), _as_array(y_pred)
    denominator = float(np.sum(np.abs(actual)))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def rmspe(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared percentage error, ignoring zero actuals.

    Kaggle's official Rossmann metric, kept so a result here is comparable with the
    public leaderboard. The competition's own rule is that zero-sales rows are excluded,
    and that exclusion is reproduced rather than reinvented.
    """
    actual, predicted = _as_array(y_true), _as_array(y_pred)
    nonzero = actual != 0
    if not np.any(nonzero):
        return float("nan")
    ratio = (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
    return float(np.sqrt(np.mean(ratio**2)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination — the share of variance the model explains.

    Reported because it is the metric the course teaches and the one a reader expects,
    and because it makes every model here comparable to the notebooks it came from.
    It is *not* the metric that decides anything, for one reason worth saying out loud:
    pooled across 1115 stores, most of the variance in `sales` is variance *between*
    stores, not within them. A model that learns only "store 262 is busy and store 307
    is quiet" scores well on this and forecasts nothing. WMAPE and MASE are scored on
    the part that is actually hard. ADR 0005.

    Delegates to scikit-learn rather than reimplementing the formula: the definition is
    not in dispute, and a hand-rolled copy is one more thing to keep correct.
    """
    return float(sk_r2_score(_as_array(y_true), _as_array(y_pred)))


def mase(y_true: ArrayLike, y_pred: ArrayLike, *, y_baseline: ArrayLike) -> float:
    """Model MAE divided by the baseline's MAE over the same window.

    Below 1 means the model beat the baseline. Exactly 1 when the model *is* the
    baseline, which the backtest asserts as a self-check.
    """
    baseline_error = mae(y_true, y_baseline)
    if baseline_error == 0.0 or np.isnan(baseline_error):
        return float("nan")
    return mae(y_true, y_pred) / baseline_error
