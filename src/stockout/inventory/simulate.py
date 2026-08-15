"""Price a forecast by the decision it drives. STUB.

This is the module that makes the project worth building rather than being row 10 of a
carousel. A forecast is not good because its WMAPE is low; it is good because stocking to
it loses less money than stocking to the alternative. Those are different orderings, and
a forecast can win on the first and lose on the second.

**The deliverable.** Run the simulator across a sweep of target service levels for each
forecaster, and plot cost against fill rate. Three curves — seasonal-naive, GBM point
with a normal safety stock, GBM quantile — on one efficient-frontier chart. Whichever
curve sits below and to the right wins, and the chart is the argument.

**The honesty note that must survive into the README.** Rossmann records store-level
revenue, not SKU units, and has no inventory column at all. Demand and stock here are
therefore both in currency units of stock-at-cost, which is internally consistent and is
*not* a unit-level simulation. Converting revenue to pseudo-units via an assumed basket
size would add decimal places and no truth. M5 is the upgrade path if real units matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import DEFAULT_LEAD_TIME_DAYS, DEFAULT_REVIEW_PERIOD_DAYS


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
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    initial_stock: float = 0.0,
) -> SimulationResult:
    """Walk the inventory forward day by day under an order-up-to policy. STUB.

    Two modelling decisions to make explicitly when implementing, and to record in
    `docs/results.md` either way:

    1. **Unmet demand is lost, not backordered.** Retail walk-outs do not queue. This
       makes shortage cost a function of the shortfall itself rather than of delay.
    2. **Deliveries arrive at the start of the day**, so stock ordered `lead_time` days
       ago is available to serve today's demand.
    """
    raise NotImplementedError


def frontier(
    demand: pd.Series,
    quantile_forecasts: pd.DataFrame,
    *,
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
) -> pd.DataFrame:
    """Cost and fill rate at each target service level, for the efficient-frontier plot. STUB."""
    raise NotImplementedError
