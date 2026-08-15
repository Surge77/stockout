"""The decision layer.

Each test here names a modelling decision that had to be made deliberately rather than
discovered by accident once numbers started coming out.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.inventory.simulate import SimulationResult, frontier, simulate


def test_the_result_totals_its_two_cost_components() -> None:
    """The dataclass fixes the shape of the answer before the answer exists."""
    result = SimulationResult(
        fill_rate=0.95,
        cycle_service_level=0.8,
        stockout_days=3,
        holding_cost=120.0,
        shortage_cost=45.0,
        mean_on_hand=210.0,
    )
    assert result.total_cost == pytest.approx(165.0)


def test_stocking_to_a_higher_quantile_trades_holding_cost_for_fill_rate() -> None:
    """The frontier's whole shape. If this does not hold, the simulator is wrong."""
    demand = pd.Series([100.0] * 60)
    low = simulate(demand, pd.Series([90.0] * 60))
    high = simulate(demand, pd.Series([130.0] * 60))
    assert high.fill_rate > low.fill_rate
    assert high.holding_cost > low.holding_cost


def test_unmet_demand_is_lost_rather_than_backordered() -> None:
    """Retail walk-outs do not queue, so a shortfall never reappears as later demand."""
    demand = pd.Series([50.0, 200.0, 50.0])
    result = simulate(demand, pd.Series([50.0, 50.0, 50.0]), initial_stock=50.0)
    assert result.stockout_days == 1


def test_a_perfect_forecast_with_ample_stock_never_stocks_out() -> None:
    demand = pd.Series([100.0] * 30)
    assert simulate(demand, pd.Series([100.0] * 30), initial_stock=1000.0).stockout_days == 0


def test_the_frontier_returns_one_row_per_service_level() -> None:
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.5": [100.0] * 60, "0.9": [130.0] * 60})
    assert len(frontier(demand, quantiles)) == 2


def test_shortage_cost_prices_the_shortfall_and_holding_prices_what_is_left() -> None:
    """Hand-checkable: 100 of demand against a level of 40, twice over.

    Day one serves 40 and loses 60; day two refills to 40 and does the same. With the
    default cost pair that is 2 x 60 x 3 short and nothing held.
    """
    result = simulate(pd.Series([100.0, 100.0]), pd.Series([40.0, 40.0]))
    assert result.shortage_cost == pytest.approx(360.0)
    assert result.holding_cost == pytest.approx(0.0)
    assert result.fill_rate == pytest.approx(0.4)
    assert result.mean_on_hand == pytest.approx(0.0)


def test_one_short_day_writes_off_the_whole_review_cycle() -> None:
    """Cycle service level is harsher than fill rate on purpose — a planner feels it that way."""
    demand = pd.Series([10.0] * 14)
    order_up_to = pd.Series([10.0] * 14)
    order_up_to.iloc[3] = 0.0

    result = simulate(demand, order_up_to, review_period_days=7)
    assert result.stockout_days == 1
    assert result.fill_rate == pytest.approx(13 / 14)
    assert result.cycle_service_level == pytest.approx(0.5)


def test_a_window_with_no_demand_is_served_perfectly_rather_than_undefined() -> None:
    empty = pd.Series([], dtype="float64")
    result = simulate(empty, empty)
    assert result.fill_rate == 1.0
    assert result.cycle_service_level == 1.0
    assert result.mean_on_hand == 0.0
    assert result.stockout_days == 0


def test_a_review_period_below_one_day_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1 day"):
        simulate(pd.Series([1.0]), pd.Series([1.0]), review_period_days=0)


def test_negative_opening_stock_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        simulate(pd.Series([1.0]), pd.Series([1.0]), initial_stock=-1.0)


def test_negative_demand_is_rejected_rather_than_treated_as_a_return() -> None:
    with pytest.raises(ValueError, match="demand cannot be negative"):
        simulate(pd.Series([-1.0]), pd.Series([1.0]))


def test_demand_and_levels_must_describe_the_same_days() -> None:
    with pytest.raises(ValueError, match="same length"):
        simulate(pd.Series([1.0, 2.0]), pd.Series([1.0]))


def test_the_frontier_needs_something_to_price() -> None:
    with pytest.raises(ValueError, match="no quantile forecasts"):
        frontier(pd.Series([1.0]), pd.DataFrame(index=[0]))


def test_the_frontier_reads_its_quantile_out_of_the_column_label() -> None:
    """`0.9` plots against the x axis; a label that is not a number becomes NaN, not a crash."""
    demand = pd.Series([100.0] * 14)
    table = frontier(demand, pd.DataFrame({"0.9": [130.0] * 14, "mean": [100.0] * 14}))
    assert table["quantile"].iloc[0] == pytest.approx(0.9)
    assert pd.isna(table["quantile"].iloc[1])


def test_a_higher_quantile_buys_fill_rate_with_carried_stock() -> None:
    """The frontier read as it is meant to be: better service, more stock, both monotone."""
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.5": [90.0] * 60, "0.9": [120.0] * 60})
    table = frontier(demand, quantiles, lead_time_days=0, review_period_days=1)

    assert table["fill_rate"].is_monotonic_increasing
    assert table["mean_on_hand"].is_monotonic_increasing
    assert table["shortage_cost"].is_monotonic_decreasing
