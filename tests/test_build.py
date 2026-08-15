"""Feature assembly, and the denylist that keeps `customers` out of the matrix."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.features import calendar
from stockout.features.build import (
    FEATURE_DENYLIST,
    assert_no_denied_columns,
    build_features,
    feature_columns,
)

HORIZON = 42


def test_customers_is_never_offered_as_a_feature(sales: pd.DataFrame) -> None:
    """It correlates with sales at ~0.9 and nobody knows it six weeks ahead.

    A model handed this column reports a wonderful score and cannot be deployed for
    a single day.
    """
    built = build_features(sales, horizon=HORIZON)
    assert s.CUSTOMERS in built.columns
    assert s.CUSTOMERS not in feature_columns(built)


def test_the_target_and_its_index_are_never_features(sales: pd.DataFrame) -> None:
    columns = feature_columns(build_features(sales, horizon=HORIZON))
    assert s.SALES not in columns
    assert s.DATE not in columns


def test_the_raw_categorical_holiday_string_is_replaced_by_a_flag(sales: pd.DataFrame) -> None:
    columns = feature_columns(build_features(sales, horizon=HORIZON))
    assert s.STATE_HOLIDAY not in columns
    assert calendar.IS_STATE_HOLIDAY in columns


def test_lag_and_rolling_columns_reach_the_feature_list(sales: pd.DataFrame) -> None:
    columns = feature_columns(build_features(sales, horizon=HORIZON))
    assert any(c.startswith("sales_lag_") for c in columns)
    assert any(c.startswith("sales_roll_mean_") for c in columns)
    assert any(c.startswith("sales_roll_std_") for c in columns)


def test_default_lags_all_clear_the_horizon(sales: pd.DataFrame) -> None:
    built = build_features(sales, horizon=HORIZON)
    lags = [int(c.rsplit("_", 1)[1]) for c in built.columns if c.startswith("sales_lag_")]
    assert lags
    assert min(lags) >= HORIZON


def test_explicit_lags_are_honoured(sales: pd.DataFrame) -> None:
    built = build_features(sales, horizon=7, lags=[7, 14])
    assert "sales_lag_7" in built.columns
    assert "sales_lag_14" in built.columns


def test_feature_columns_are_all_numeric(sales: pd.DataFrame) -> None:
    built = build_features(sales, horizon=HORIZON)
    for column in feature_columns(built):
        assert pd.api.types.is_numeric_dtype(built[column])


def test_a_hand_assembled_feature_list_is_checked(sales: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown at the forecast origin"):
        assert_no_denied_columns([s.PROMO, s.CUSTOMERS])


def test_a_clean_hand_assembled_list_passes() -> None:
    assert_no_denied_columns([s.PROMO, s.STORE])


def test_the_denylist_names_the_columns_it_claims_to() -> None:
    assert {s.SALES, s.DATE, s.CUSTOMERS, s.STATE_HOLIDAY} <= FEATURE_DENYLIST


def test_building_features_does_not_mutate_its_input(sales: pd.DataFrame) -> None:
    before = sales.copy()
    build_features(sales, horizon=HORIZON)
    pd.testing.assert_frame_equal(sales, before)
