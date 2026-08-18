"""The four arms, and the one property that makes the comparison mean anything."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.loaders import merge_store
from stockout.data.synth import make_sales
from stockout.data.synth_stores import make_stores
from stockout.evaluate.leakage import ARM_COLUMNS, SMUGGLED_CUSTOMERS, Arm, leakage_arms
from stockout.features.build import FEATURE_DENYLIST, build_features, feature_columns
from stockout.features.store_features import add_store_features
from stockout.split.strategies import date_major

HORIZON = 7
TEST_DAYS = 28


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    built = merge_store(make_sales(n_stores=4, days=500), make_stores(n_stores=4))
    built = build_features(add_store_features(built), horizon=HORIZON)
    return date_major(built).dropna(subset=["sales_lag_28"])


@pytest.fixture(scope="module")
def arms(frame: pd.DataFrame) -> pd.DataFrame:
    return leakage_arms(frame, test_days=TEST_DAYS, gap_days=HORIZON)


def test_every_arm_is_reported(arms: pd.DataFrame) -> None:
    assert list(arms.columns) == list(ARM_COLUMNS)
    assert set(arms["arm"]) == {
        "honest",
        "shuffled split",
        "preprocessing leak",
        "future feature",
    }


def test_optimism_is_the_gap_between_the_two_scores(arms: pd.DataFrame) -> None:
    """The column the module exists to produce, so it is worth checking it is that."""
    assert (arms["optimism"] - (arms["internal"] - arms["future"])).abs().max() < 1e-12


def test_every_arm_is_scored_on_the_same_future_window(frame: pd.DataFrame) -> None:
    """The property the whole design rests on.

    Comparing protocols by their own scores varies training size, test period and
    leakage at once. Holding the future fixed leaves only what each protocol *believes*
    about itself, which is the thing being measured.
    """
    from stockout.split.strategies import time_holdout

    _, future = time_holdout(frame, test_days=TEST_DAYS, gap_days=HORIZON)
    assert future[s.DATE].nunique() == TEST_DAYS


def test_a_future_feature_scores_near_perfectly_and_is_still_useless(
    arms: pd.DataFrame,
) -> None:
    """`customers` correlates with sales at about 0.9 and does not exist on the day you
    forecast. A high held-out score is not evidence that a model can be deployed."""
    smuggled = arms.loc[arms["arm"] == "future feature", "future"].iloc[0]
    honest = arms.loc[arms["arm"] == "honest", "future"].iloc[0]
    assert smuggled > honest
    assert smuggled > 0.99


def test_the_denylist_has_to_be_lied_to_before_the_leak_can_be_shown() -> None:
    """The strongest evidence the guard is load-bearing: bypassing it needs a rename."""
    assert s.CUSTOMERS in FEATURE_DENYLIST
    assert SMUGGLED_CUSTOMERS not in FEATURE_DENYLIST


def test_the_smuggled_column_is_picked_up_as_a_feature(frame: pd.DataFrame) -> None:
    renamed = frame.assign(**{SMUGGLED_CUSTOMERS: frame[s.CUSTOMERS]})
    assert SMUGGLED_CUSTOMERS in feature_columns(renamed)
    assert s.CUSTOMERS not in feature_columns(renamed)


def test_the_honest_arm_does_not_flatter_itself(arms: pd.DataFrame) -> None:
    """Its optimism should be small. Not zero — the validation and future windows are
    different stretches of calendar — but small, and that is the control."""
    honest = arms.loc[arms["arm"] == "honest", "optimism"].iloc[0]
    assert abs(honest) < 0.15


def test_an_arm_computes_its_own_optimism() -> None:
    arm = Arm(arm="x", protocol="p", features="f", internal=0.94, future=0.81)
    assert arm.optimism == pytest.approx(0.13)
    assert arm.as_row()["optimism"] == pytest.approx(0.13)


def test_a_window_longer_than_the_data_is_rejected(frame: pd.DataFrame) -> None:
    from stockout.errors import BacktestError

    with pytest.raises(BacktestError):
        leakage_arms(frame, test_days=10_000, gap_days=0)
