"""The conformal arithmetic, against numbers written by hand.

No model, no fitting, no LightGBM. Everything here has one right answer that can be
worked out on paper, which is exactly why it was split out of `conformal.py` — an
off-by-one in an order statistic is a wrong service level, and it should fail as an
arithmetic assertion rather than as a slightly different third decimal place six modules
downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.models.conformity import (
    SCALE_FLOOR,
    conformal_quantile,
    floored_scale,
    grouped_offsets,
    min_rows_for,
    pooled_offsets,
    saturates,
)


def test_the_correction_is_the_order_statistic_and_not_a_plain_percentile() -> None:
    """`ceil((n + 1) * level) / n` is what makes the coverage a guarantee.

    Ten scores at 0.9: `ceil(11 * 0.9) / 10 = 1.0`, so the largest is taken. A plain
    90th percentile would take the ninth, under-covering by exactly the correction the
    finite-sample term exists to supply.
    """
    scores = np.arange(10, dtype="float64")
    assert conformal_quantile(scores, 0.9) == pytest.approx(9.0)


def test_the_correction_saturates_rather_than_running_off_the_end() -> None:
    """With few scores the corrected level exceeds 1 and must clamp, not raise."""
    assert conformal_quantile(np.array([1.0, 2.0]), 0.95) == pytest.approx(2.0)


def test_saturation_is_detected_from_the_row_count_alone() -> None:
    """The row count a level needs is derived, not chosen.

    `ceil((n + 1) * level) < n` is the condition for the correction to land strictly
    inside the sample. For a 0.99 that first holds at 199 rows, and for a 0.9 at 19 —
    which is why a 42-day calibration window over a handful of stores can estimate the
    middle of the grid and not the top of it.
    """
    assert saturates(198, 0.99)
    assert not saturates(199, 0.99)

    assert saturates(18, 0.9)
    assert not saturates(19, 0.9)

    assert saturates(0, 0.5)


def test_the_scale_floor_keeps_a_zero_prediction_out_of_the_denominator() -> None:
    """A shut store predicts zero, and an infinity in a pooled quantile is not a number."""
    assert SCALE_FLOOR > 0.0
    assert floored_scale(pd.Series([0.0, -5.0, 100.0])).to_numpy() == pytest.approx(
        [SCALE_FLOOR, SCALE_FLOOR, 100.0]
    )


def test_the_offset_is_relative_so_it_travels_between_a_quiet_shop_and_a_busy_one() -> None:
    """The reason the score is divided by the prediction rather than pooled raw.

    Two rows, one selling ten times the other, each under-forecast by exactly half. A raw
    pooled residual would be 50 on one row and 500 on the other and would fit neither; in
    units of the prediction both are 1.0, and one offset serves both.
    """
    frame = pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-01-01", periods=2, freq="D"),
            s.STORE: [1, 2],
            s.SALES: [100.0, 1000.0],
            s.OPEN: [1, 1],
        }
    )
    predicted = pd.DataFrame({"0.5": [50.0, 500.0]})

    offsets = pooled_offsets(
        predicted=predicted, scale=pd.Series([50.0, 500.0]), frame=frame
    )
    assert offsets[0.5] == pytest.approx(1.0)


# --- per-group calibration (ADR 0012) -------------------------------------------------


def test_the_row_floor_is_derived_from_the_strictest_level_asked_for() -> None:
    """Not a chosen constant: the floor is whatever `saturates` says it has to be.

    A grid topping out at 0.5 needs 3 rows, at 0.9 needs 19, at 0.99 needs 199. Asking
    for a higher service level is what makes per-group calibration expensive, and the
    cost is arithmetic rather than a matter of taste.
    """
    assert min_rows_for((0.5,)) == 3
    assert min_rows_for((0.5, 0.9)) == 19
    assert min_rows_for((0.5, 0.9, 0.99)) == 199

    for level in (0.5, 0.75, 0.9, 0.95, 0.99):
        floor = min_rows_for((level,))
        assert not saturates(floor, level)
        assert saturates(floor - 1, level)


def test_a_level_at_or_above_one_has_no_floor_and_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        min_rows_for((0.5, 1.0))


def _grouped_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Two groups. `wide` sells 100 on twenty days, `thin` sells 300 on four."""
    sales = [100.0] * 20 + [300.0] * 4
    groups = ["wide"] * 20 + ["thin"] * 4
    frame = pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-01-01", periods=24, freq="D"),
            s.STORE: groups,
            s.SALES: sales,
            s.OPEN: [1] * 24,
        }
    )
    predicted = pd.DataFrame({"0.5": [40.0] * 24})
    return frame, predicted, pd.Series(groups)


def test_a_group_with_enough_rows_is_corrected_on_its_own_residuals() -> None:
    """The point of Mondrian: `wide` is short by 60 and `thin` by 260, at a scale of 1."""
    frame, predicted, groups = _grouped_frame()

    offsets, thin = grouped_offsets(
        predicted=predicted,
        scale=pd.Series([1.0] * 24),
        frame=frame,
        groups=groups,
        min_rows=3,
    )

    assert thin == ()
    assert offsets["wide"][0.5] == pytest.approx(60.0)
    assert offsets["thin"][0.5] == pytest.approx(260.0)


def test_a_group_below_the_floor_gets_no_offset_and_is_named() -> None:
    """No entry rather than a noisy one, so the caller substitutes the marginal offset."""
    frame, predicted, groups = _grouped_frame()

    offsets, thin = grouped_offsets(
        predicted=predicted,
        scale=pd.Series([1.0] * 24),
        frame=frame,
        groups=groups,
        min_rows=19,
    )

    assert thin == ("thin",)
    assert list(offsets) == ["wide"]


def test_closed_days_do_not_count_towards_a_group_clearing_the_floor() -> None:
    """A shut Sunday is not a residual here either, so it cannot buy a group its own offset."""
    frame, predicted, groups = _grouped_frame()
    frame.loc[frame[s.STORE] == "thin", s.OPEN] = 0

    offsets, thin = grouped_offsets(
        predicted=predicted,
        scale=pd.Series([1.0] * 24),
        frame=frame,
        groups=groups,
        min_rows=3,
    )

    assert thin == ()
    assert list(offsets) == ["wide"]


def test_the_grouping_has_to_describe_the_calibration_rows() -> None:
    frame, predicted, _ = _grouped_frame()
    with pytest.raises(ValueError, match="same length"):
        grouped_offsets(
            predicted=predicted,
            scale=pd.Series([1.0] * 24),
            frame=frame,
            groups=pd.Series(["wide"]),
            min_rows=3,
        )
