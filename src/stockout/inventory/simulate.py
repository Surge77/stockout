"""Price a forecast by the decision it drives.

This is the module that makes the project worth building rather than being row 10 of a
carousel. A forecast is not good because its WMAPE is low; it is good because stocking to
it loses less money than stocking to the alternative. Those are different orderings, and
a forecast can win on the first and lose on the second.

**The deliverable.** Run the simulator across a sweep of target service levels for each
forecaster, and plot cost against fill rate. Three curves — seasonal-naive, GBM point
with a normal safety stock, GBM quantile — on one efficient-frontier chart. Whichever
curve sits below and to the right wins, and the chart is the argument.

**What is simulated by default is a repeated newsvendor.** Stock is raised to the
order-up-to level at the start of every day, demand arrives, and whatever was not served
walks out. With `lead_time_days=0` the delivery is instant, so the protection interval is
one day — which makes each day a single-period stocking decision, which is exactly the
problem `policy.critical_ratio` solves. The cost-minimising quantile should therefore land
near `Cu / (Cu + Co)`, and that is a claim the frontier can be checked against rather than
a framing chosen for convenience.

**Pass a lead time and the loop becomes an `(R, S)` system.** An order placed on day `t`
arrives on day `t + L`; until then it is in transit, counted in the inventory position and
absent from the shelf. The two halves of the model must move together — a pipeline needs a
level sized across the protection interval, and `frontier` builds one with
`policy.order_up_to_level`. Doing only one of the two is the trap ADR 0008 documents and
this repository walked into once: a level sized to survive a wait, refilled daily by an
instant delivery, is permanent overstock, and every service level saturates at a fill rate
of 1.0 with no trade-off left to see. ADR 0010 records the pipeline and what it costs.

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
    "mean_on_order",
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

    mean_on_order: float = 0.0
    """Mean stock in transit. Zero without a lead time, and the evidence of one with it."""

    @property
    def total_cost(self) -> float:
        return self.holding_cost + self.shortage_cost


def simulate(
    demand: pd.Series,
    order_up_to: pd.Series,
    *,
    lead_time_days: int = 0,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    initial_stock: float = 0.0,
    holding_cost: float = DEFAULT_OVERAGE_COST,
    shortage_cost: float = DEFAULT_UNDERAGE_COST,
) -> SimulationResult:
    """Walk the inventory forward day by day under an order-up-to policy.

    Three modelling decisions, made explicitly:

    1. **Unmet demand is lost, not backordered.** Retail walk-outs do not queue, so a
       shortfall never reappears as tomorrow's demand. Shortage cost is therefore a
       function of the shortfall itself rather than of any delay in clearing it.
    2. **The order is placed against the inventory position**, on-hand plus in-transit,
       and never against on-hand alone. Re-ordering stock already on a lorry is the
       textbook bullwhip, and `policy.order_quantity` names the same rule.
    3. **`lead_time_days=0` is a same-day delivery**, which collapses the loop to the
       repeated newsvendor of ADR 0008: position equals on-hand, the order lands
       immediately, and the protection interval is the single day. That is the default
       because it is the decision the newsvendor critical ratio actually solves. Pass a
       lead time to open the pipeline and price an `(R, S)` system instead — ADR 0010,
       and note that `order_up_to` must then be a *protection-interval* level rather than
       one day's quantile, or the shelf will be permanently short.

    Costs are per unit: `holding_cost` for each unit still on the shelf at the end of a
    day, `shortage_cost` for each unit of demand that walked out unserved.
    """
    if len(demand) != len(order_up_to):
        raise ValueError("demand and order-up-to levels must be the same length")
    if lead_time_days < 0:
        raise ValueError("lead time cannot be negative")
    if review_period_days < 1:
        raise ValueError("review period must be at least 1 day")
    if initial_stock < 0:
        raise ValueError("initial stock cannot be negative")

    wanted = demand.to_numpy(dtype="float64")
    levels = order_up_to.to_numpy(dtype="float64")
    if bool((wanted < 0).any()):
        raise ValueError("demand cannot be negative")

    # One slot per simulated day plus the tail an order placed on the last day would
    # arrive into. Those trailing slots are never read, and allocating them is cheaper
    # than a bounds check inside the loop.
    arrivals = np.zeros(len(wanted) + lead_time_days + 1, dtype="float64")
    on_hand = float(initial_stock)
    on_order = 0.0
    served_total = holding_total = shortage_total = 0.0
    end_of_day: list[float] = []
    in_transit: list[float] = []
    was_short: list[bool] = []

    for day, (want, level) in enumerate(zip(wanted, levels, strict=True)):
        arriving = arrivals[day]
        on_hand += arriving
        on_order -= arriving

        ordered = max(level - (on_hand + on_order), 0.0)
        if lead_time_days == 0:
            on_hand += ordered
        else:
            arrivals[day + lead_time_days] += ordered
            on_order += ordered

        served = min(on_hand, want)
        on_hand -= served
        shortfall = want - served

        served_total += served
        holding_total += on_hand * holding_cost
        shortage_total += shortfall * shortage_cost
        end_of_day.append(on_hand)
        in_transit.append(on_order)
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
        mean_on_order=float(np.mean(in_transit)) if in_transit else 0.0,
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
