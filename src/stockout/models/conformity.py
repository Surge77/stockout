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
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..data import schemas as s
from ..errors import BacktestError

#: Lower bound on the per-row scale the conformity score is divided by, in the currency
#: units the target is measured in. A shut store predicts zero, and dividing a residual
#: by zero would put an infinity into a pooled quantile; the floor makes the arithmetic
#: total without changing any row that carries real demand.
SCALE_FLOOR = 1.0


def split_calibration_window(
    train: pd.DataFrame, *, days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner training window and calibration tail, divided by date and never by row.

    Dividing by row would put some of a day's stores on one side and the rest on the
    other, which is a random split wearing a timestamp.

    Both degenerate cases are refused here, before any model is fitted, because finding
    out after training two sets of boosters that the window was unusable wastes the
    fitting and arrives as a worse message.
    """
    dates = pd.to_datetime(train[s.DATE])
    cutoff = pd.Timestamp(dates.max()) - pd.Timedelta(days=days)
    inner = train[dates <= cutoff]
    calibration = train[dates > cutoff]

    if inner.empty:
        raise BacktestError(
            f"a {days}-day calibration window leaves no training data; the training "
            f"frame spans {len(dates.unique())} days"
        )
    if not bool((calibration[s.OPEN] == 1).any()):
        raise BacktestError(
            "the calibration window contains no trading day, so no residual can be "
            "measured; lengthen it with calibration_days"
        )
    return inner, calibration


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


def min_rows_for(levels: Iterable[float]) -> int:
    """The row count the strictest level asked for needs before its offset is an estimate.

    Derived from `saturates` rather than chosen: the answer is the smallest `n` at which
    `ceil((n + 1) * q) < n` for the highest `q` in the grid — 3 rows for a 0.5, 19 for a
    0.9, 199 for a 0.99. That makes the floor on per-group calibration a consequence of
    the grid the caller asked for, not a constant somebody liked the look of. ADR 0009
    refused a magic threshold and this is the alternative it implied.
    """
    strictest = max(levels)
    if not 0.0 < strictest < 1.0:
        raise ValueError("quantile levels must lie strictly between 0 and 1")

    # The answer is *defined* as the smallest n where `saturates` is False, so it is found
    # by asking `saturates` rather than by algebra. `(1 + q) / (1 - q)` is the closed form
    # and it cannot be trusted to the integer: at q=0.9 the division lands on
    # 19.000000000000004 and ceiling it gives 20, one row past the true floor and off by
    # one in the unsafe direction. Truncating instead undershoots, which the scan repairs.
    n = max(1, int((1.0 + strictest) / (1.0 - strictest)) - 1)
    while saturates(n, strictest):
        n += 1
    return n


def grouped_offsets(
    *,
    predicted: pd.DataFrame,
    scale: pd.Series,
    frame: pd.DataFrame,
    groups: pd.Series,
    min_rows: int,
) -> tuple[dict[object, dict[float, float]], tuple[object, ...]]:
    """A separate pooled offset per group — Mondrian conformal — where the rows allow it.

    Returns the offsets of the groups that cleared `min_rows` trading rows, and the names
    of the groups that did not. A group below the floor gets no entry at all rather than a
    noisy one: the caller substitutes the marginal offset, and knows which groups it did
    that for, because a correction that quietly degrades into a different correction is
    worse than one that says so.

    Splitting the residual pool is not free. Every group is estimated from a fraction of
    the rows, so what is bought in conditional validity is paid for in variance, and below
    the floor the payment exceeds the purchase — which is the finding ADR 0009 recorded at
    70 pooled rows and the reason this has a floor at all. ADR 0012.
    """
    trading = np.asarray(frame[s.OPEN] == 1)
    labels = np.asarray(groups)
    if labels.size != trading.size:
        raise ValueError("the grouping and the calibration frame must be the same length")

    offsets: dict[object, dict[float, float]] = {}
    thin: list[object] = []
    for name in sorted(set(labels[trading].tolist())):
        inside = trading & (labels == name)
        if int(inside.sum()) < min_rows:
            thin.append(name)
            continue
        offsets[name] = pooled_offsets(
            predicted=predicted.loc[inside],
            scale=scale.loc[inside],
            frame=frame.loc[inside],
        )
    return offsets, tuple(thin)


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


def offsets_for_rows(
    *,
    labels: pd.Series,
    group_offsets: dict[object, dict[float, float]],
    pooled: float,
    level: float,
) -> pd.Series:
    """Per-row correction: the row's own group offset where one exists, the pooled one else."""
    return labels.map(lambda name: group_offsets.get(name, {}).get(level, pooled))


def groups_without_offsets(
    labels: pd.Series, group_offsets: dict[object, dict[float, float]]
) -> tuple[object, ...]:
    """Labels present in `labels` that no group offset covers, so the caller can say so."""
    return tuple(sorted(set(labels[~labels.isin(list(group_offsets))].tolist())))


def floored_scale(reference: pd.Series) -> pd.Series:
    """Per-row divisor for the conformity score: the median prediction, floored."""
    return reference.clip(lower=SCALE_FLOOR)
