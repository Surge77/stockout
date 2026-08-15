"""Price a forecast by the decision it drives.

This is the module that makes the project worth building rather than being row 10 of a
carousel. A forecast is not good because its WMAPE is low; it is good because stocking to
it loses less money than stocking to the alternative. Those are different orderings, and
a forecast can win on the first and lose on the second.

**The deliverable.** Run the simulator across a sweep of target service levels for each
forecaster, and plot cost against fill rate. Three curves — seasonal-naive, GBM point
with a normal safety stock, GBM quantile — on one efficient-frontier chart. Whichever
curve sits below and to the right wins, and the chart is the argument.

**How replenishment is timed, and what that costs.** Stock is raised to the order-up-to
level at the start of every day, with no shipping lag inside the loop. The lead time is
not ignored — it is already inside the *level*, which `policy.order_up_to_level` builds by
summing the forecast across the lead time plus one review period. Applying it again here
would charge the protection interval twice. What this does bias is holding cost: a real
(R, S) system lets stock cycle down between deliveries and this one refills daily, so
carried stock is overstated. Every curve on the frontier carries the same bias, which is
why the chart is read as an ordering of curves and not as a budget. Recorded in ADR 0008.

**The honesty note that must survive into the README.** Rossmann records store-level
revenue, not SKU units, and has no inventory column at all. Demand and stock here are
therefore both in currency units of stock-at-cost, which is internally consistent and is
*not* a unit-level simulation. Converting revenue to pseudo-units via an assumed basket
size would add decimal places and no truth. M5 is the upgrade path if real units matter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_OVERAGE_COST,
    DEFAULT_REVIEW_PERIOD_DAYS,
    DEFAULT_UNDERAGE_COST,
)
from .policy import order_up_to_level

FRONTIER_COLUMNS: tuple[str, ...] = (
    "quantile",
    "fill_rate",
    "cycle_service_level",
    "stockout_days",
    "holding_cost",
    "shortage_cost",
    "total_cost",
    "mean_on_hand",
)


@dataclass(frozen=True)
class SimulationResult:
    """What one policy run cost, and how often it failed the customer."""

    fill_rate: float
    """Share of demand met from stock. The service measure that matters to a customer."""

    cycle_service_level: float
    """Share of review cycles with no stockout at all. What planners usually quote."""

    stockout_days: int
    holding_cost: float
    shortage_cost: float
    mean_on_hand: float

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost


def simulate(
    demand: pd.Series,
    order_up_to: pd.Series,
    *,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    initial_stock: float = 0.0,
    holding_cost: float = DEFAULT_OVERAGE_COST,
    shortage_cost: float = DEFAULT_UNDERAGE_COST,
) -> SimulationResult:
    """Walk the inventory forward day by day under an order-up-to policy.

    Two modelling decisions, made explicitly:

    1. **Unmet demand is lost, not backordered.** Retail walk-outs do not queue, so a
       shortfall never reappears as tomorrow's demand. Shortage cost is therefore a
       function of the shortfall itself rather than of any delay in clearing it.
    2. **Stock is topped up at the start of each day**, by the `order_quantity` rule —
       raise to S, never send stock back. The protection interval that S covers is the
       caller's business; see the module docstring for why it is not re-applied here.

    Costs are per unit: `holding_cost` for each unit still on the shelf at the end of a
    day, `shortage_cost` for each unit of demand that walked out unserved.
    """
    if len(demand) != len(order_up_to):
        raise ValueError("demand and order-up-to levels must be the same length")
    if review_period_days < 1:
        raise ValueError("review period must be at least 1 day")
    if initial_stock < 0:
        raise ValueError("initial stock cannot be negative")

    wanted = demand.to_numpy(dtype="float64")
    levels = order_up_to.to_numpy(dtype="float64")
    if bool((wanted < 0).any()):
        raise ValueError("demand cannot be negative")

    on_hand = float(initial_stock)
    served_total = holding_total = shortage_total = 0.0
    end_of_day: list[float] = []
    was_short: list[bool] = []

    for want, level in zip(wanted, levels, strict=True):
        on_hand += max(level - on_hand, 0.0)
        served = min(on_hand, want)
        on_hand -= served
        shortfall = want - served

        served_total += served
        holding_total += on_hand * holding_cost
        shortage_total += shortfall * shortage_cost
        end_of_day.append(on_hand)
        was_short.append(shortfall > 0.0)

    wanted_total = float(wanted.sum())
    return SimulationResult(
        # Demand that never arrived cannot have gone unmet, so an empty window is a
        # perfect one rather than an undefined one.
        fill_rate=served_total / wanted_total if wanted_total > 0.0 else 1.0,
        cycle_service_level=_cycle_service_level(was_short, review_period_days),
        stockout_days=sum(was_short),
        holding_cost=holding_total,
        shortage_cost=shortage_total,
        mean_on_hand=float(np.mean(end_of_day)) if end_of_day else 0.0,
    )


def frontier(
    demand: pd.Series,
    quantile_forecasts: pd.DataFrame,
    *,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    initial_stock: float = 0.0,
) -> pd.DataFrame:
    """Cost and fill rate at each target service level, for the efficient-frontier plot.

    One row per column of `quantile_forecasts`, in the order the columns arrive. Each
    column is read as a per-day demand quantile and turned into a base-stock level by
    `order_up_to_level`, which is where the lead time enters — once.
    """
    if quantile_forecasts.shape[1] == 0:
        raise ValueError("no quantile forecasts to price")

    rows = []
    for name in quantile_forecasts.columns:
        level = order_up_to_level(
            quantile_forecasts[name],
            lead_time_days=lead_time_days,
            review_period_days=review_period_days,
        )
        result = simulate(
            demand,
            level,
            review_period_days=review_period_days,
            initial_stock=initial_stock,
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
            }
        )
    return pd.DataFrame(rows, columns=list(FRONTIER_COLUMNS))


def _cycle_service_level(was_short: Sequence[bool], review_period_days: int) -> float:
    """Share of review cycles that got through without a single short day.

    Harsher than the fill rate on purpose: one bad afternoon writes off the whole cycle,
    which is how a planner experiences it.
    """
    if not was_short:
        return 1.0
    cycles = range(0, len(was_short), review_period_days)
    clean = sum(not any(was_short[start : start + review_period_days]) for start in cycles)
    return clean / len(cycles)


def _as_quantile(name: object) -> float:
    """Column label back to a number, so the frontier can be plotted against it."""
    try:
        return float(str(name))
    except ValueError:
        return float("nan")
