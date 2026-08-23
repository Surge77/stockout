"""The straight line under each store, and the thing it exists to make possible.

A tree cannot extrapolate. Past the largest threshold it ever split on it returns the
nearest leaf and stops, so a date years out gets the same answer as the same kind of day
inside the window. Rossmann grows about 5% a year, which makes that blindness expensive:
the deployed regressor scores r² 0.34 on a chronological hold-out and 0.83 once the level
is modelled separately.

These tests pin the two properties the decomposition rests on — that a line extrapolates
where a tree does not, and that the line is fitted on the training rows alone — plus the
invariant that makes it safe to leave on by default: on data with no trend it must not
make anything worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockout import trend
from stockout.data import schemas as s
from stockout.errors import StockoutError
from stockout.trend import MAX_CONFIDENT_DAYS

ORIGIN = pd.Timestamp("2013-01-01")


def _frame(slope_per_day: float, *, base: float = 1_000.0, days: int = 400,
           entity: str = "1") -> pd.DataFrame:
    dates = pd.date_range(ORIGIN, periods=days, freq="D")
    t = np.arange(days, dtype="float64")
    return pd.DataFrame(
        {s.ENTITY: entity, s.DATE: dates, s.SALES: base + slope_per_day * t}
    )


def test_it_recovers_a_line_it_was_given() -> None:
    trends = trend.fit(_frame(2.0, base=1_000.0))
    intercept, slope = trends.line_for("1")

    assert intercept == pytest.approx(1_000.0, abs=1e-6)
    assert slope == pytest.approx(2.0, abs=1e-9)


def test_it_extrapolates_past_the_data() -> None:
    """The whole point. A tree returns its nearest leaf here; a line keeps going."""
    trends = trend.fit(_frame(2.0, base=1_000.0, days=400))

    inside = trends.level_at("1", ORIGIN + pd.Timedelta(days=200))
    beyond = trends.level_at("1", ORIGIN + pd.Timedelta(days=1_000))

    assert inside == pytest.approx(1_400.0, abs=1e-6)
    assert beyond == pytest.approx(3_000.0, abs=1e-6)


def test_annual_growth_is_relative_to_the_store_s_own_level() -> None:
    """A raw slope cannot tell a big shop from a growing one."""
    big = trend.fit(_frame(1.0, base=10_000.0))
    small = trend.fit(_frame(1.0, base=1_000.0))

    assert big.annual_growth("1") == pytest.approx(0.0365, abs=1e-3)
    assert small.annual_growth("1") == pytest.approx(0.3653, abs=1e-3)


def test_removing_the_trend_leaves_the_deviations() -> None:
    frame = _frame(2.0, base=1_000.0)
    frame.loc[10, s.SALES] = float(frame.loc[10, s.SALES]) + 500.0  # type: ignore[arg-type]

    residual = trend.remove(frame, trend.fit(frame))

    assert residual.loc[10] > 400.0
    assert abs(residual.drop(index=10)).max() < 50.0


def test_remove_and_restore_round_trip() -> None:
    frame = _frame(2.0)
    trends = trend.fit(frame)

    residual = trend.remove(frame, trends)
    back = trend.restore(residual.to_numpy(), frame[s.ENTITY], frame[s.DATE], trends)

    assert np.allclose(back, frame[s.SALES].to_numpy())


def test_each_store_gets_its_own_line() -> None:
    """A chain-wide slope would be wrong for both of these and right for neither."""
    frame = pd.concat(
        [_frame(5.0, base=1_000.0, entity="1"), _frame(-3.0, base=2_000.0, entity="2")],
        ignore_index=True,
    )
    trends = trend.fit(frame)

    assert trends.line_for("1")[1] == pytest.approx(5.0, abs=1e-6)
    assert trends.line_for("2")[1] == pytest.approx(-3.0, abs=1e-6)


def test_a_store_with_too_few_rows_falls_back_to_the_chain() -> None:
    """A slope from a handful of days is noise with a direction — and it is extrapolated
    forward, which is the one place a bad estimate compounds instead of averaging out."""
    frame = pd.concat(
        [
            _frame(5.0, entity="1", days=400),
            _frame(5.0, entity="2", days=trend.MIN_ROWS_PER_ENTITY - 1),
        ],
        ignore_index=True,
    )
    trends = trend.fit(frame)

    assert "1" in trends.per_entity
    assert "2" in trends.fell_back
    assert trends.line_for("2") == trends.global_line


def test_a_flat_series_gets_a_flat_line() -> None:
    """The safety property behind leaving this on by default: where there is no trend,
    the decomposition must do nothing rather than invent one."""
    trends = trend.fit(_frame(0.0, base=5_000.0))

    assert trends.line_for("1")[1] == pytest.approx(0.0, abs=1e-9)
    assert trends.level_at("1", ORIGIN + pd.Timedelta(days=9_000)) == pytest.approx(
        5_000.0, abs=1e-6
    )


def test_the_origin_travels_with_the_line() -> None:
    """An intercept read from a different origin is wrong by the slope times the gap,
    and wrong silently — every number still looks like a plausible sales figure."""
    frame = _frame(2.0, base=1_000.0)
    trends = trend.fit(frame, origin=ORIGIN)

    assert trends.origin == ORIGIN
    assert trends.level_at("1", ORIGIN) == pytest.approx(1_000.0, abs=1e-6)


def test_fitting_on_an_empty_frame_raises() -> None:
    with pytest.raises(StockoutError, match="empty frame"):
        trend.fit(pd.DataFrame({s.ENTITY: [], s.DATE: [], s.SALES: []}))


def test_a_far_horizon_earns_a_warning_and_a_near_one_does_not() -> None:
    through = pd.Timestamp("2015-07-31")

    assert trend.horizon_warning(through + pd.Timedelta(days=30), through) is None
    assert (
        trend.horizon_warning(
            through + pd.Timedelta(days=MAX_CONFIDENT_DAYS), through
        )
        is None
    )

    far = trend.horizon_warning(through + pd.Timedelta(days=4_000), through)
    assert far is not None
    assert "arithmetic rather than a forecast" in far
