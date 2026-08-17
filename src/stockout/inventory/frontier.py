"""Sweep the service levels and price every one of them.

**The deliverable.** Run the simulator across a sweep of target service levels for each
forecaster, and plot cost against fill rate. Three curves — seasonal-naive, GBM point
with a normal safety stock, GBM quantile — on one efficient-frontier chart. Whichever
curve sits below and to the right wins, and the chart is the argument.

One `simulate` call per quantile column, which is why this lives next to the day loop
rather than inside it: `simulate` prices one policy, and the interesting object is the
shape traced out by pricing all of them. Cost is U-shaped in the service level and
accuracy is monotone in it, so "as accurate as possible" and "as cheap as possible" are
different instructions — that difference is the whole project, and it is visible only in
the sweep.
"""

from __future__ import annotations

import pandas as pd

from ..config import (
    DEFAULT_OVERAGE_COST,
    DEFAULT_REVIEW_PERIOD_DAYS,
    DEFAULT_UNDERAGE_COST,
)
from .policy import order_up_to_level
from .simulate import simulate

FRONTIER_COLUMNS: tuple[str, ...] = (
    "quantile",
    "fill_rate",
    "cycle_service_level",
    "stockout_days",
    "holding_cost",
    "shortage_cost",
    "total_cost",
    "mean_on_hand",
    "mean_on_order",
)


def frontier(
    demand: pd.Series,
    quantile_forecasts: pd.DataFrame,
    *,
    lead_time_days: int = 0,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    initial_stock: float | None = None,
    holding_cost: float = DEFAULT_OVERAGE_COST,
    shortage_cost: float = DEFAULT_UNDERAGE_COST,
) -> pd.DataFrame:
    """Cost and fill rate at each target service level, for the efficient-frontier plot.

    One row per column of `quantile_forecasts`, in the order the columns arrive.

    Without a lead time each column is that day's demand quantile and is stocked to
    directly, because the decision being priced is a single-period one. With one, the
    same column is summed forward across the protection interval by
    `policy.order_up_to_level` and the loop opens a delivery pipeline — the two changes
    belong together, and applying either alone is the mistake ADR 0008 documents.

    The review period for *sizing* is one day, because this loop reviews daily. The
    `review_period_days` argument groups days into cycles for `cycle_service_level` and
    still does not change what is ordered.

    `initial_stock` defaults to opening in steady state: nothing when deliveries are
    instant, and the first day's base-stock level when they are not. A pipeline that
    starts empty guarantees a stockout on every day before the first lorry arrives, and
    charging a policy for the warehouse having been built yesterday measures the opening
    balance rather than the policy.
    """
    if quantile_forecasts.shape[1] == 0:
        raise ValueError("no quantile forecasts to price")
    if lead_time_days < 0:
        raise ValueError("lead time cannot be negative")

    rows = []
    for name in quantile_forecasts.columns:
        levels = _levels_for(quantile_forecasts[name], lead_time_days=lead_time_days)
        opening = _opening_stock(levels, initial_stock, lead_time_days=lead_time_days)
        result = simulate(
            demand,
            levels,
            lead_time_days=lead_time_days,
            review_period_days=review_period_days,
            initial_stock=opening,
            holding_cost=holding_cost,
            shortage_cost=shortage_cost,
        )
        rows.append(
            {
                "quantile": _as_quantile(name),
                "fill_rate": result.fill_rate,
                "cycle_service_level": result.cycle_service_level,
                "stockout_days": result.stockout_days,
                "holding_cost": result.holding_cost,
                "shortage_cost": result.shortage_cost,
                "total_cost": result.total_cost,
                "mean_on_hand": result.mean_on_hand,
                "mean_on_order": result.mean_on_order,
            }
        )
    return pd.DataFrame(rows, columns=list(FRONTIER_COLUMNS))


def _levels_for(forecast: pd.Series, *, lead_time_days: int) -> pd.Series:
    """The day's own quantile, or its forward sum across the protection interval."""
    if lead_time_days == 0:
        return forecast
    return order_up_to_level(forecast, lead_time_days=lead_time_days, review_period_days=1)


def _opening_stock(
    levels: pd.Series, initial_stock: float | None, *, lead_time_days: int
) -> float:
    """The caller's number, or a shelf that starts where the policy would keep it.

    Only a pipeline needs the warm start. With instant delivery the first day is topped
    up before demand arrives, so an empty opening shelf costs nothing and saying zero is
    plainer than saying something that cancels.
    """
    if initial_stock is not None:
        return initial_stock
    if lead_time_days == 0 or levels.empty:
        return 0.0
    return float(levels.iloc[0])


def _as_quantile(name: object) -> float:
    """Column label back to a number, so the frontier can be plotted against it."""
    try:
        return float(str(name))
    except ValueError:
        return float("nan")
