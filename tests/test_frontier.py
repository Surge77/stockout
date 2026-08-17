"""The service-level sweep.

`test_simulate.py` checks that one policy is priced correctly. These check that pricing
*all* of them produces a curve with a trade-off in it — which is the claim the project
makes, and the one it got wrong once already (ADR 0008).
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.inventory.frontier import frontier


def test_the_frontier_returns_one_row_per_service_level() -> None:
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.5": [100.0] * 60, "0.9": [130.0] * 60})
    assert len(frontier(demand, quantiles)) == 2


def test_the_frontier_needs_something_to_price() -> None:
    with pytest.raises(ValueError, match="no quantile forecasts"):
        frontier(pd.Series([1.0]), pd.DataFrame(index=[0]))


def test_the_frontier_reads_its_quantile_out_of_the_column_label() -> None:
    """`0.9` plots against the x axis; a label that is not a number becomes NaN, not a crash."""
    demand = pd.Series([100.0] * 14)
    table = frontier(demand, pd.DataFrame({"0.9": [130.0] * 14, "mean": [100.0] * 14}))
    assert table["quantile"].iloc[0] == pytest.approx(0.9)
    assert pd.isna(table["quantile"].iloc[1])


def test_the_frontier_stocks_to_the_forecast_itself_not_to_a_multi_day_cover() -> None:
    """Guards the mistake that flattened the first frontier this repo produced.

    A base-stock level sized to survive a lead time, refilled every day, is permanent
    overstock: every service level saturates at a fill rate of 1.0 and there is no
    trade-off left to plot. Stocking to the day's own quantile is what makes the curve
    a curve.
    """
    demand = pd.Series([100.0] * 30)
    table = frontier(demand, pd.DataFrame({"0.5": [50.0] * 30}))
    assert table["fill_rate"].iloc[0] == pytest.approx(0.5)


def test_a_higher_quantile_buys_fill_rate_with_carried_stock() -> None:
    """The frontier read as it is meant to be: better service, more stock, both monotone."""
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.5": [90.0] * 60, "0.9": [120.0] * 60})
    table = frontier(demand, quantiles, review_period_days=1)

    assert table["fill_rate"].is_monotonic_increasing
    assert table["mean_on_hand"].is_monotonic_increasing
    assert table["shortage_cost"].is_monotonic_decreasing


def test_a_negative_lead_time_is_rejected_before_anything_is_priced() -> None:
    with pytest.raises(ValueError, match="lead time cannot be negative"):
        frontier(pd.Series([1.0]), pd.DataFrame({"0.9": [1.0]}), lead_time_days=-1)


def test_the_frontier_sizes_across_the_protection_interval_once_deliveries_take_time() -> None:
    """A lead time must move the level too, or the shelf starves for the wrong reason.

    Stocking to a single day's quantile under a seven-day wait would leave every service
    level short; summing the quantile across the protection interval is what makes the
    curve a curve again.
    """
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.5": [90.0] * 60, "0.9": [120.0] * 60})
    table = frontier(demand, quantiles, lead_time_days=7, review_period_days=1)

    assert (table["mean_on_order"] > 0.0).all()
    assert table["fill_rate"].is_monotonic_increasing
    assert table["fill_rate"].iloc[0] < 1.0


def test_a_pipeline_opens_in_steady_state_rather_than_on_an_empty_shelf() -> None:
    """Charging a policy for the warehouse having been built yesterday measures nothing.

    With an explicitly empty opening shelf the first week is lost outright; with the
    default the same policy is judged on the days it actually controls.
    """
    demand = pd.Series([100.0] * 60)
    quantiles = pd.DataFrame({"0.9": [120.0] * 60})

    warm = frontier(demand, quantiles, lead_time_days=7, review_period_days=1)
    cold = frontier(demand, quantiles, lead_time_days=7, review_period_days=1, initial_stock=0.0)

    assert cold["stockout_days"].iloc[0] > warm["stockout_days"].iloc[0]
