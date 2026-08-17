"""The conformal arithmetic, apart from the forecaster that applies it.

Split out of `conformal.py` because the two answer different questions. This module asks
"given these residuals, what is the correction?" — a question with one right answer,
checkable against numbers written by hand. `conformal.py` asks "which model's residuals,
measured on which window, and what happens to a shut store?" — a question about
modelling, checkable only against a fitted model.

Everything here is pure: no fitting, no state, nothing that needs LightGBM. That is the
point. Most of what can go wrong in a calibration layer is arithmetic — the wrong order
statistic, a missing finite-sample correction, closed days pooled into the residuals —
and none of it should need a booster to expose.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data import schemas as s

#: Lower bound on the per-row scale the conformity score is divided by, in the currency
#: units the target is measured in. A shut store predicts zero, and dividing a residual
#: by zero would put an infinity into a pooled quantile; the floor makes the arithmetic
#: total without changing any row that carries real demand.
SCALE_FLOOR = 1.0


def pooled_offsets(
    *, predicted: pd.DataFrame, scale: pd.Series, frame: pd.DataFrame
) -> dict[float, float]:
    """One conformity correction per quantile column, in units of the scale.

    Pooled across every trading row in the window, which is what makes the resulting
    coverage statement a marginal one: correct on average, and silent about any
    particular store or month. `conformal.py` records that as a cost and
    `metrics.coverage_by_segment` is how it stops being invisible.

    Closed days are excluded for the reason they are excluded everywhere else: a zero on
    a shut Sunday is not a demand observation, and a residual of exactly zero repeated
    across a seventh of the window drags every pooled quantile toward the middle.

    That the window holds at least one trading row is the caller's guarantee, checked
    before either model is fitted rather than after both are.
    """
    trading = np.asarray(frame[s.OPEN] == 1)
    actual = np.asarray(frame[s.SALES], dtype="float64")[trading]
    divisor = np.asarray(floored_scale(scale), dtype="float64")[trading]

    return {
        float(column): conformal_quantile(
            (actual - np.asarray(predicted[column], dtype="float64")[trading]) / divisor,
            float(column),
        )
        for column in predicted.columns
    }


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """The `ceil((n + 1) * level) / n` order statistic of `scores`.

    The `n + 1` is the finite-sample correction that makes the coverage statement hold
    for a calibration set of this size rather than for an infinite one, and
    `method="higher"` rounds towards more coverage rather than interpolating between two
    neighbours. Both err towards covering.

    Whether the statement is a theorem or an expectation depends on what the caller did
    with the model afterwards, not on this function — see ADR 0009 and ADR 0013.
    """
    n = int(scores.size)
    corrected = min(math.ceil((n + 1) * level) / n, 1.0)
    return float(np.quantile(scores, corrected, method="higher"))


def saturates(n: int, level: float) -> bool:
    """True when the corrected level runs off the end of the residuals there are.

    The correction lands strictly inside the sample only when `ceil((n + 1) * level) < n`,
    which for a 0.99 first holds at 199 rows and for a 0.9 at 19. Below that the offset is
    not an estimate of a quantile, it is the worst thing that happened once — a fact about
    the calibration window rather than about the model. It is reported instead of being
    smoothed over, because a service level resting on a single observation should say so.
    """
    return n < 1 or math.ceil((n + 1) * level) / n >= 1.0


def floored_scale(reference: pd.Series) -> pd.Series:
    """Per-row divisor for the conformity score: the median prediction, floored."""
    return reference.clip(lower=SCALE_FLOOR)
