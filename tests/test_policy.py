"""The newsvendor critical ratio — implemented — and the policy stubs around it."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.inventory.policy import critical_ratio, order_quantity, order_up_to_level


def test_the_default_costs_imply_a_seventy_five_percent_service_level() -> None:
    """A lost sale hurting three times as much as carried stock gives 3/(3+1)."""
    assert critical_ratio() == pytest.approx(0.75)


@pytest.mark.parametrize(
    "underage, overage, expected",
    [(1.0, 1.0, 0.5), (9.0, 1.0, 0.9), (1.0, 9.0, 0.1), (5.0, 0.0, 1.0), (0.0, 5.0, 0.0)],
)
def test_the_ratio_moves_with_the_cost_pair(
    underage: float, overage: float, expected: float
) -> None:
    """The service level is derived from the business, not tuned by a grid search."""
    assert critical_ratio(underage_cost=underage, overage_cost=overage) == pytest.approx(expected)


def test_negative_costs_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        critical_ratio(underage_cost=-1.0)


def test_two_zero_costs_have_no_defined_optimum() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        critical_ratio(underage_cost=0.0, overage_cost=0.0)


@pytest.mark.skip(reason="stub — implement in P4, see docs/decisions/0007")
def test_the_protection_interval_covers_lead_time_plus_one_review_period() -> None:
    """Covering only the lead time understocks by a whole review period.

    Stock ordered now must also carry the store until the *next* order can arrive.
    """
    forecast = pd.Series([100.0] * 14)
    level = order_up_to_level(forecast, lead_time_days=7, review_period_days=7)
    assert level.iloc[0] == pytest.approx(1400.0)


@pytest.mark.skip(reason="stub — implement in P4")
def test_order_quantity_never_goes_negative() -> None:
    """Overstocked stores order nothing; they do not send stock back."""
    quantity = order_quantity(pd.Series([500.0]), pd.Series([300.0]))
    assert quantity.iloc[0] == 0.0


@pytest.mark.skip(reason="stub — implement in P4")
def test_order_quantity_counts_stock_already_in_transit() -> None:
    """Counting on-hand only re-orders everything in transit each cycle — a bullwhip."""
    on_hand, on_order = 100.0, 200.0
    quantity = order_quantity(pd.Series([on_hand + on_order]), pd.Series([400.0]))
    assert quantity.iloc[0] == pytest.approx(100.0)
