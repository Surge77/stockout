"""Replenishment policy: the service level, the base-stock level, and the order.

**The point.** "Which quantile should I forecast?" looks like a hyperparameter and is
not. For a single-period stocking decision the optimal service level is the newsvendor
critical ratio `Cu / (Cu + Co)` — the cost of being one unit short over the cost of being
one unit short plus one unit long. It falls out of the cost pair, so it is derived, not
tuned, and it moves when the business changes rather than when a grid search is rerun.

With the defaults (a lost sale hurts three times as much as carrying stock) the answer is
0.75, and the forecaster should be fitting the 0.75 quantile rather than the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_OVERAGE_COST,
    DEFAULT_REVIEW_PERIOD_DAYS,
    DEFAULT_UNDERAGE_COST,
)


def critical_ratio(
    *,
    underage_cost: float = DEFAULT_UNDERAGE_COST,
    overage_cost: float = DEFAULT_OVERAGE_COST,
) -> float:
    """The newsvendor optimum: `Cu / (Cu + Co)`. This is the quantile to forecast."""
    if underage_cost < 0 or overage_cost < 0:
        raise ValueError("costs cannot be negative")
    total = underage_cost + overage_cost
    if total == 0:
        raise ValueError("at least one of the costs must be non-zero")
    return underage_cost / total


def order_up_to_level(
    quantile_forecast: pd.Series,
    *,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
) -> pd.Series:
    """Base-stock level S covering demand over the lead time plus one review period.

    The protection interval is `lead_time + review_period`, not the lead time alone —
    stock ordered now must also cover the gap until the *next* order can arrive. Getting
    this wrong understocks by a whole review period and is the classic textbook error.

    The window looks forward: S at day t sums the forecast for days t through
    `t + protection - 1`. At the end of the series there are fewer days left to cover, so
    the level tapers instead of extrapolating demand nobody forecast — the alternative is
    inventing figures in order to make a chart end tidily.
    """
    if lead_time_days < 0:
        raise ValueError("lead time cannot be negative")
    if review_period_days < 1:
        raise ValueError("review period must be at least 1 day")

    protection = lead_time_days + review_period_days
    # Reversed so that pandas' trailing window becomes a leading one. A forward-looking
    # sum over a *forecast* is not leakage: every value in it is a prediction, not an
    # observation.
    backwards = quantile_forecast.iloc[::-1]
    return backwards.rolling(protection, min_periods=1).sum().iloc[::-1]


def order_quantity(
    inventory_position: pd.Series,
    order_up_to: pd.Series,
) -> pd.Series:
    """`max(S - inventory_position, 0)`, where position counts stock on hand plus on order.

    Counting only on-hand stock re-orders everything already in transit every review
    period, which produces a textbook bullwhip.

    Compared positionally rather than by index label. Two series that describe the same
    days but were sliced differently would otherwise align into silent NaNs.
    """
    if len(inventory_position) != len(order_up_to):
        raise ValueError("inventory position and order-up-to levels must be the same length")

    shortfall = order_up_to.to_numpy(dtype="float64") - inventory_position.to_numpy(
        dtype="float64"
    )
    return pd.Series(np.maximum(shortfall, 0.0), index=order_up_to.index)
