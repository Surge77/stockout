"""Lag and rolling features, and the guard that proves they cannot see forward.

`test_features_ignore_a_perturbed_final_target` is the test this repository exists for.
It does not check a formula; it checks a property — that changing the newest target value
changes no feature anywhere. Any leak, however it is written, breaks it.

Its partner, `test_perturbing_an_old_target_does_change_later_features`, exists so the
first test cannot pass by being vacuous. A feature builder that returned constants would
satisfy the sentinel and fail this one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.errors import LeakageError
from stockout.features.build import build_features, feature_columns
from stockout.features.lags import add_lags, add_rolling, seasonal_lags

HORIZON = 42


def test_features_ignore_a_perturbed_final_target(sales: pd.DataFrame) -> None:
    original = build_features(sales, horizon=HORIZON)

    tampered = sales.copy()
    tampered.loc[tampered.index[-1], s.SALES] = 1e9
    perturbed = build_features(tampered, horizon=HORIZON)

    columns = feature_columns(original)
    pd.testing.assert_frame_equal(original[columns], perturbed[columns])


def test_perturbing_an_old_target_does_change_later_features(sales: pd.DataFrame) -> None:
    original = build_features(sales, horizon=HORIZON)

    tampered = sales.copy()
    tampered.loc[tampered.index[0], s.SALES] = 1e9
    perturbed = build_features(tampered, horizon=HORIZON)

    columns = feature_columns(original)
    assert not original[columns].equals(perturbed[columns])


def test_lag_shorter_than_horizon_raises(tiny: pd.DataFrame) -> None:
    with pytest.raises(LeakageError, match="shorter than the 42-day horizon"):
        add_lags(tiny, lags=[1, 7], horizon=42)


def test_lag_equal_to_horizon_is_allowed(tiny: pd.DataFrame) -> None:
    out = add_lags(tiny, lags=[2], horizon=2)
    assert "sales_lag_2" in out.columns


def test_lags_do_not_bleed_across_stores(tiny: pd.DataFrame) -> None:
    out = add_lags(tiny, lags=[1], horizon=1)
    first_row_of_second_store = out[out[s.STORE] == 2].iloc[0]
    assert pd.isna(first_row_of_second_store["sales_lag_1"])


def test_rolling_window_excludes_the_target_row(tiny: pd.DataFrame) -> None:
    """Store 1 runs 100..105. At the fourth row, a horizon-1 window of 3 covers 100-102."""
    out = add_rolling(tiny, windows=[3], horizon=1)
    store_one = out[out[s.STORE] == 1].reset_index(drop=True)
    assert store_one.loc[3, "sales_roll_mean_3"] == pytest.approx(101.0)
    assert store_one.loc[3, s.SALES] == 103.0


def test_rolling_respects_the_horizon_shift(tiny: pd.DataFrame) -> None:
    """At horizon 2 the same row may only see up to 101, so a 2-wide mean is 100.5."""
    out = add_rolling(tiny, windows=[2], horizon=2)
    store_one = out[out[s.STORE] == 1].reset_index(drop=True)
    assert store_one.loc[3, "sales_roll_mean_2"] == pytest.approx(100.5)


def test_rolling_rejects_non_positive_windows(tiny: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="positive"):
        add_rolling(tiny, windows=[0], horizon=1)


def test_lags_reject_non_positive_values(tiny: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="positive"):
        add_lags(tiny, lags=[0], horizon=1)


def test_horizon_must_be_at_least_one_day(tiny: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least 1 day"):
        add_lags(tiny, lags=[7], horizon=0)


@pytest.mark.parametrize(
    "horizon, expected",
    [
        (42, [42, 49, 56, 63]),
        (43, [49, 56, 63, 70]),
        (1, [7, 14, 21, 28]),
        (7, [7, 14, 21, 28]),
    ],
)
def test_seasonal_lags_land_on_whole_weeks_clearing_the_horizon(
    horizon: int, expected: list[int]
) -> None:
    assert seasonal_lags(horizon) == expected


def test_seasonal_lags_rejects_zero_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        seasonal_lags(42, count=0)
