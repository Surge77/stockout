"""The decision layer.

Each test here names a modelling decision that had to be made deliberately rather than
discovered by accident once numbers started coming out.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.inventory.simulate import SimulationResult, simulate


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


# --- the delivery pipeline (ADR 0010) -------------------------------------------------


def test_without_a_lead_time_nothing_is_ever_in_transit() -> None:
    """The default is still the repeated newsvendor of ADR 0008, to the last decimal."""
    result = simulate(pd.Series([100.0] * 10), pd.Series([120.0] * 10))
    assert result.mean_on_order == pytest.approx(0.0)


def test_an_order_placed_today_cannot_be_sold_today() -> None:
    """The whole content of a lead time, in four days.

    Nothing is wanted until the last day, by which point the order placed on the first
    has landed. Move the demand to the front and the same policy is short, because the
    lorry has not arrived yet.
    """
    level = pd.Series([100.0] * 4)
    late = simulate(pd.Series([0.0, 0.0, 0.0, 100.0]), level, lead_time_days=2)
    early = simulate(pd.Series([100.0, 0.0, 0.0, 0.0]), level, lead_time_days=2)

    assert late.stockout_days == 0
    assert early.stockout_days == 1


def test_stock_already_on_a_lorry_is_not_ordered_a_second_time() -> None:
    """The bullwhip guard, hand-checkable.

    Ten quiet days, a level of 100 and a three-day wait. Exactly one order of 100 is
    placed, on day one, and it sits in transit for three days before landing. A loop that
    compared the level against on-hand alone would order another 100 every morning and
    have 300 in the air by the third day.
    """
    result = simulate(
        pd.Series([0.0] * 10), pd.Series([100.0] * 10), lead_time_days=3, review_period_days=1
    )
    assert result.mean_on_order == pytest.approx(30.0)
    assert result.mean_on_hand == pytest.approx(70.0)


def test_a_single_day_level_under_a_multi_day_wait_starves_the_shelf() -> None:
    """The other half of ADR 0008's trap, and the reason `frontier` re-sizes.

    A level that covers one day of demand, ordered against a three-day lead time, can
    never hold more than one day of cover and spends most of the window empty. The
    pipeline is not a free upgrade: it has to be paid for in the level.
    """
    result = simulate(
        pd.Series([100.0] * 12), pd.Series([100.0] * 12), lead_time_days=3, review_period_days=1
    )
    assert result.stockout_days >= 8
    assert result.fill_rate < 0.5


def test_a_negative_lead_time_is_rejected_rather_than_read_as_early_delivery() -> None:
    with pytest.raises(ValueError, match="lead time cannot be negative"):
        simulate(pd.Series([1.0]), pd.Series([1.0]), lead_time_days=-1)
