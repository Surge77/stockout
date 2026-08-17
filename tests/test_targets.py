"""What Low, Medium and High mean, and the traps in saying so."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.config import DEMAND_CLASS_LABELS
from stockout.data import schemas as s
from stockout.data.synth import make_sales
from stockout.errors import SchemaError
from stockout.targets import (
    DEMAND_CLASS,
    DEMAND_CLASS_CODE,
    DemandThresholds,
    absolute_thresholds,
    add_demand_class,
    class_balance,
    encode_labels,
    fit_thresholds,
    labelled_rows,
)


def _two_stores() -> pd.DataFrame:
    """A quiet store and a busy one, with no overlap in their sales ranges.

    The non-overlap is the point: under global cut points every quiet day is Low and
    every busy day is High, so the two labelling schemes disagree on every single row.
    """
    quiet = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    busy = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    return pd.DataFrame(
        {
            s.STORE: [1] * 6 + [2] * 6,
            s.SALES: quiet + busy,
            s.OPEN: [1] * 12,
            s.DATE: list(pd.date_range("2014-01-01", periods=6)) * 2,
        }
    )


# --- per store, not globally --------------------------------------------------------------


def test_each_store_gets_its_own_cut_points() -> None:
    thresholds = fit_thresholds(_two_stores())
    assert thresholds.for_store(1) != thresholds.for_store(2)
    assert thresholds.stores == 2


def test_per_store_terciles_spread_a_quiet_store_across_all_three_classes() -> None:
    """The whole reason for the design: a quiet shop still has busy days."""
    labelled = add_demand_class(_two_stores(), fit_thresholds(_two_stores()))
    quiet = labelled[labelled[s.STORE] == 1]
    assert set(quiet[DEMAND_CLASS]) == set(DEMAND_CLASS_LABELS)


def test_absolute_thresholds_deny_a_quiet_store_its_busy_days() -> None:
    """Which is why the absolute version scores better and is worth less.

    Under one global pair of cut points the quiet store never reaches High and the busy
    store never reaches Low, whatever either of them actually did that day. A classifier
    trained on these labels is being graded on recognising which store it is looking at,
    and it will score well for it.
    """
    labelled = add_demand_class(_two_stores(), absolute_thresholds(_two_stores()))
    quiet = set(labelled.loc[labelled[s.STORE] == 1, DEMAND_CLASS])
    busy = set(labelled.loc[labelled[s.STORE] == 2, DEMAND_CLASS])
    assert "High" not in quiet
    assert "Low" not in busy


def test_the_training_window_is_split_into_roughly_equal_thirds() -> None:
    frame = make_sales(n_stores=4, days=365)
    balance = class_balance(add_demand_class(frame, fit_thresholds(frame)))
    assert balance.min() > 0.30
    assert balance.max() < 0.37


# --- fitted on training rows only -----------------------------------------------------------


def test_cut_points_come_from_the_frame_they_are_fitted_on() -> None:
    """Refitting on the test window would let it vote on its own labels."""
    frame = make_sales(n_stores=2, days=400)
    train = frame[frame[s.DATE] < "2013-09-01"]
    assert fit_thresholds(train).for_store(1) != fit_thresholds(frame).for_store(1)


def test_a_store_absent_from_training_falls_back_to_the_pooled_cut_points() -> None:
    """Roughly a sixth of Rossmann's stores vanish for a refurbishment quarter."""
    thresholds = fit_thresholds(_two_stores())
    assert thresholds.for_store(999) == thresholds.fallback


def test_a_training_window_with_no_trading_days_is_rejected() -> None:
    shut = _two_stores().assign(**{s.OPEN: 0, s.SALES: 0.0})
    with pytest.raises(SchemaError, match="no trading days"):
        fit_thresholds(shut)


# --- closed days ------------------------------------------------------------------------------


def test_a_closed_day_gets_no_class_rather_than_the_bottom_one() -> None:
    """'Low' must not come to mean 'shut'. It is a question nobody asked."""
    frame = _two_stores()
    frame.loc[0, s.OPEN] = 0
    frame.loc[0, s.SALES] = 0.0
    labelled = add_demand_class(frame, fit_thresholds(_two_stores()))
    assert pd.isna(labelled[DEMAND_CLASS_CODE].iloc[0])


def test_only_labelled_trading_days_are_offered_for_training() -> None:
    frame = make_sales(n_stores=2, days=200)
    labelled = add_demand_class(frame, fit_thresholds(frame))
    trainable = labelled_rows(labelled)
    assert len(trainable) == int((frame[s.OPEN] == 1).sum())
    assert trainable[DEMAND_CLASS_CODE].notna().all()


def test_closed_days_do_not_move_the_cut_points() -> None:
    """A closure mass at zero would drag the lower tercile down onto it."""
    trading_only = _two_stores()
    with_closures = pd.concat(
        [trading_only, trading_only.assign(**{s.OPEN: 0, s.SALES: 0.0})], ignore_index=True
    )
    assert fit_thresholds(with_closures).for_store(1) == fit_thresholds(trading_only).for_store(1)


# --- the ordering trap --------------------------------------------------------------------------


def test_the_class_codes_run_low_to_high_and_not_alphabetically() -> None:
    """`LabelEncoder` sorts, so it would give High=0, Low=1, Medium=2 — and every
    confusion-matrix caption would be wrong while looking entirely plausible."""
    assert list(encode_labels(pd.Series(["Low", "Medium", "High"]))) == [0, 1, 2]


def test_the_categorical_carries_the_same_order_as_the_codes() -> None:
    labelled = add_demand_class(_two_stores(), fit_thresholds(_two_stores()))
    assert list(labelled[DEMAND_CLASS].cat.categories) == list(DEMAND_CLASS_LABELS)
    assert labelled[DEMAND_CLASS].cat.ordered


def test_the_label_and_its_code_always_agree() -> None:
    """Two representations of one fact, so they are worth checking against each other."""
    labelled = labelled_rows(add_demand_class(_two_stores(), fit_thresholds(_two_stores())))
    expected = encode_labels(labelled[DEMAND_CLASS].astype(str))
    assert list(labelled[DEMAND_CLASS_CODE]) == list(expected)


def test_a_label_outside_the_three_classes_is_rejected() -> None:
    with pytest.raises(SchemaError, match="Low/Medium/High"):
        encode_labels(pd.Series(["Low", "Enormous"]))


# --- boundaries and errors -----------------------------------------------------------------------


def test_a_value_exactly_on_a_cut_point_falls_to_the_lower_class() -> None:
    """np.quantile lands on an observed value whenever the count divides by three."""
    thresholds = DemandThresholds(per_store={1: (100.0, 200.0)}, fallback=(0.0, 0.0))
    frame = pd.DataFrame({s.STORE: [1, 1], s.SALES: [100.0, 200.0], s.OPEN: [1, 1]})
    assert list(add_demand_class(frame, thresholds)[DEMAND_CLASS]) == ["Low", "Medium"]


def test_labelling_a_frame_without_sales_is_rejected() -> None:
    with pytest.raises(SchemaError, match="demand classes need"):
        fit_thresholds(pd.DataFrame({s.STORE: [1]}))


def test_asking_for_a_balance_before_labelling_says_so() -> None:
    with pytest.raises(SchemaError, match="add_demand_class"):
        class_balance(_two_stores())


def test_asking_for_trainable_rows_before_labelling_says_so() -> None:
    with pytest.raises(SchemaError, match="add_demand_class"):
        labelled_rows(_two_stores())


def test_the_input_frame_is_not_mutated() -> None:
    frame = _two_stores()
    add_demand_class(frame, fit_thresholds(frame))
    assert DEMAND_CLASS not in frame.columns
