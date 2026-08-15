"""Price a forecast by the decision it drives.

This is the module that makes the project worth building rather than being row 10 of a
carousel. A forecast is not good because its WMAPE is low; it is good because stocking to
it loses less money than stocking to the alternative. Those are different orderings, and
a forecast can win on the first and lose on the second.

**The deliverable.** Run the simulator across a sweep of target service levels for each
forecaster, and plot cost against fill rate. Three curves — seasonal-naive, GBM point
with a normal safety stock, GBM quantile — on one efficient-frontier chart. Whichever
curve sits below and to the right wins, and the chart is the argument.

**What is being simulated is a repeated newsvendor.** Stock is raised to the order-up-to
level at the start of every day, demand arrives, and whatever was not served walks out.
There is no shipping lag inside the loop, so the protection interval is one day — which
makes each day a single-period stocking decision, which is exactly the problem
`policy.critical_ratio` solves. The cost-minimising quantile should therefore land near
`Cu / (Cu + Co)`, and that is a claim the frontier can be checked against rather than a
framing chosen for convenience.

The multi-period `(R, S)` system — order every `R` days, wait `L` for delivery, size `S`
to cover `L + R` — is a different model, and `policy.order_up_to_level` builds its level.
Feeding that level to this loop is the one thing not to do: a level sized to survive a
wait, refilled daily, is permanent overstock, and every service level saturates at a fill
rate of 1.0 with no trade-off left to see. Recorded in ADR 0008, with the measurement.

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
    DEFAULT_OVERAGE_COST,
    DEFAULT_REVIEW_PERIOD_DAYS,
    DEFAULT_UNDERAGE_COST,
)

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
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    initial_stock: float = 0.0,
    holding_cost: float = DEFAULT_OVERAGE_COST,
    shortage_cost: float = DEFAULT_UNDERAGE_COST,
) -> pd.DataFrame:
    """Cost and fill rate at each target service level, for the efficient-frontier plot.

    One row per column of `quantile_forecasts`, in the order the columns arrive. Each
    column is that day's demand quantile and is stocked to directly, because the decision
    being priced is a single-period one; see the module docstring for why a multi-period
    base-stock level does not belong here.

    `review_period_days` groups days into cycles for `cycle_service_level` and does not
    change what is ordered. The cost pair is the same one `policy.critical_ratio` derives
    the service level from, and passing it rather than reading a constant is what keeps
    the two consistent when a caller overrides it.
    """
    if quantile_forecasts.shape[1] == 0:
        raise ValueError("no quantile forecasts to price")

    rows = []
    for name in quantile_forecasts.columns:
        result = simulate(
            demand,
            quantile_forecasts[name],
            review_period_days=review_period_days,
            initial_stock=initial_stock,
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
