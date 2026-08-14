"""The decision layer. Every test here is skipped until P4 builds it.

These are written now, before the implementation, because they are the specification.
Each one names a modelling decision that has to be made deliberately rather than
discovered by accident once numbers start coming out.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.inventory.simulate import SimulationResult, frontier, simulate


def test_the_result_totals_its_two_cost_components() -> None:
    """Not a stub: the dataclass is real, so the shape of the answer is already fixed."""
    result = SimulationResult(
        fill_rate=0.95,
        cycle_service_level=0.8,
        stockout_days=3,
        holding_cost=120.0,
        shortage_cost=45.0,
        mean_on_hand=210.0,
    )
    assert result.total_cost == pytest.approx(165.0)


@pytest.mark.skip(reason="stub — implement in P4")
def test_stocking_to_a_higher_quantile_trades_holding_cost_for_fill_rate() -> None:
    """The frontier's whole shape. If this does not hold, the simulator is wrong."""
    demand = pd.Series([100.0] * 60)
    low = simulate(demand, pd.Series([90.0] * 60))
    high = simulate(demand, pd.Series([130.0] * 60))
    assert high.fill_rate > low.fill_rate
    assert high.holding_cost > low.holding_cost


@pytest.mark.skip(reason="stub — implement in P4")
def test_unmet_demand_is_lost_rather_than_backordered() -> None:
    """Retail walk-outs do not queue, so a shortfall never reappears as later demand."""
    demand = pd.Series([50.0, 200.0, 50.0])
    result = simulate(demand, pd.Series([50.0, 50.0, 50.0]), initial_stock=50.0)
    assert result.stockout_days == 1


@pytest.mark.skip(reason="stub — implement in P4")
def test_a_perfect_forecast_with_ample_stock_never_stocks_out() -> None:
    demand = pd.Series([100.0] * 30)
    assert simulate(demand, pd.Series([100.0] * 30), initial_stock=1000.0).stockout_days == 0


@pytest.mark.skip(reason="stub — implement in P4")
def test_the_frontier_returns_one_row_per_service_level() -> None:
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.5": [100.0] * 60, "0.9": [130.0] * 60})
    assert len(frontier(demand, quantiles)) == 2
