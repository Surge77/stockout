"""Calibration: the arithmetic against a stub, then the claim against real boosters.

Most of what can go wrong in a conformal layer is arithmetic — the wrong order statistic,
a missing finite-sample correction, closed days pooled into the residuals — and none of
it needs LightGBM to expose. `FlatQuantileModel` returns predictions with a known answer
already in them, so a wrong offset is a failed assertion rather than a slightly different
third decimal place.

The two tests at the bottom fit the real thing, because a calibration layer that works on
a stub and not on a booster has calibrated nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.errors import BacktestError
from stockout.evaluate.metrics import coverage
from stockout.models import FORECASTER_NAMES, forecaster
from stockout.models.conformal import (
    SCALE_FLOOR,
    ConformalQuantileForecaster,
    _conformal_quantile,
    _saturates,
)
from stockout.split.rolling import rolling_origin, split_frame

HORIZON = 42


class FlatQuantileModel:
    """Predicts the same number for every row, whatever it is fitted on.

    Useless as a forecaster and ideal as a fixture: the residual for every row is
    `sales - constant`, so the offset a correct implementation must produce can be
    computed by hand from the fixture's own sales column.
    """

    name = "flat"

    def __init__(self, *, levels: dict[str, float]) -> None:
        self.levels = levels
        self.quantiles = tuple(float(q) for q in levels)
        self.crossing_rate = 0.0
        self.fitted_rows = 0

    def fit(self, train: pd.DataFrame) -> FlatQuantileModel:
        self.fitted_rows = len(train)
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=future.index)

    def predict_quantiles(self, future: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {name: pd.Series(value, index=future.index) for name, value in self.levels.items()},
            index=future.index,
        )


def _flat(**levels: float) -> ConformalQuantileForecaster:
    """A calibrator wrapping a flat model, with a scale of 1 so offsets read directly."""
    return ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=10,
        factory=lambda: FlatQuantileModel(levels=levels),
    )


def _frame(sales: list[float], *, open_flags: list[int] | None = None) -> pd.DataFrame:
    """One store, consecutive days, with `sales` as given."""
    flags = [1] * len(sales) if open_flags is None else open_flags
    return pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-01-01", periods=len(sales), freq="D"),
            s.STORE: 1,
            s.SALES: sales,
            s.OPEN: flags,
        }
    )


def test_the_correction_is_the_order_statistic_and_not_a_plain_percentile() -> None:
    """`ceil((n + 1) * level) / n` is what makes the coverage a guarantee.

    Ten scores at 0.9: `ceil(11 * 0.9) / 10 = 1.0`, so the largest is taken. A plain
    90th percentile would take the ninth, under-covering by exactly the correction the
    finite-sample term exists to supply.
    """
    scores = np.arange(10, dtype="float64")
    assert _conformal_quantile(scores, 0.9) == pytest.approx(9.0)


def test_the_correction_saturates_rather_than_running_off_the_end() -> None:
    """With few scores the corrected level exceeds 1 and must clamp, not raise."""
    assert _conformal_quantile(np.array([1.0, 2.0]), 0.95) == pytest.approx(2.0)


def test_saturation_is_detected_from_the_row_count_alone() -> None:
    """The row count a level needs is derived, not chosen.

    `ceil((n + 1) * level) < n` is the condition for the correction to land strictly
    inside the sample. For a 0.99 that first holds at 199 rows, and for a 0.9 at 19 —
    which is why a 42-day calibration window over a handful of stores can estimate the
    middle of the grid and not the top of it.
    """
    assert _saturates(198, 0.99)
    assert not _saturates(199, 0.99)

    assert _saturates(18, 0.9)
    assert not _saturates(19, 0.9)

    assert _saturates(0, 0.5)


def test_a_level_that_rests_on_one_observation_is_named_rather_than_smoothed_over() -> None:
    """Ten calibration rows cannot estimate a 0.99, and the model has to admit which."""
    model = _flat(**{"0.5": 40.0, "0.99": 40.0}).fit(_frame([100.0] * 30))

    assert model.calibration_rows == 10
    assert model.saturated_quantiles == (0.99,)


def test_the_offset_is_the_residual_the_calibration_window_actually_showed() -> None:
    """Hand-checkable end to end.

    The last ten days are the calibration window and they sell 100 a day. A flat model
    predicting 40 is short by 60 on every one of them, so the offset at any level is 60
    divided by a scale of 1.
    """
    model = _flat(**{"0.9": 40.0}).fit(_frame([100.0] * 30))
    assert model.offsets[0.9] == pytest.approx(60.0)


def test_a_model_already_covering_its_level_is_pulled_back_down() -> None:
    """Calibration is not a safety margin. Over-coverage costs holding cost, so it goes."""
    model = _flat(**{"0.9": 250.0}).fit(_frame([100.0] * 30))
    assert model.offsets[0.9] == pytest.approx(-150.0)


def test_closed_days_are_kept_out_of_the_residual_pool() -> None:
    """A shut store's zero is not a demand observation here either.

    Eight of the ten calibration days are closures selling nothing, against a flat
    prediction of 40. Pooling them in would fill the pool with residuals of -40 and drag
    the median offset negative; the two trading days say +60, and +60 is the answer.
    """
    sales = [100.0] * 20 + [0.0] * 8 + [100.0] * 2
    open_flags = [1] * 20 + [0] * 8 + [1] * 2
    model = _flat(**{"0.5": 40.0}).fit(_frame(sales, open_flags=open_flags))

    assert model.calibration_rows == 2
    assert model.offsets[0.5] == pytest.approx(60.0)


def test_a_calibration_window_of_only_closed_days_is_refused_not_guessed() -> None:
    frame = _frame([100.0] * 30, open_flags=[1] * 20 + [0] * 10)
    with pytest.raises(BacktestError, match="no trading day"):
        _flat(**{"0.9": 40.0}).fit(frame)


def test_a_calibration_window_longer_than_the_history_is_refused() -> None:
    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=90,
        factory=lambda: FlatQuantileModel(levels={"0.9": 1.0}),
    )
    with pytest.raises(BacktestError, match="leaves no training data"):
        calibrator.fit(_frame([100.0] * 30))


def test_the_deployed_model_is_fitted_on_everything_including_the_calibration_tail() -> None:
    """The probe gives up data so the offsets are honest; the deployed model does not."""
    built: list[FlatQuantileModel] = []

    def factory() -> FlatQuantileModel:
        model = FlatQuantileModel(levels={"0.9": 40.0})
        built.append(model)
        return model

    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON, calibration_days=10, factory=factory
    )
    calibrator.fit(_frame([100.0] * 30))

    probe, deployed = built
    assert probe.fitted_rows == 20
    assert deployed.fitted_rows == 30


def test_predictions_are_shifted_by_the_offset_they_were_calibrated_with() -> None:
    model = _flat(**{"0.9": 40.0}).fit(_frame([100.0] * 30))
    predicted = model.predict_quantiles(_frame([0.0] * 5))
    assert predicted["0.9"].to_numpy() == pytest.approx([100.0] * 5)


def test_a_calibrated_quantile_is_never_pushed_below_zero() -> None:
    """Negative revenue is not a stocking decision, whatever the residuals said.

    The deployed model is moved after calibration so that its raw prediction is far
    smaller than the correction learned against it. That cannot happen to a flat model on
    its own, and it happens constantly to a booster asked about a quiet week.
    """
    built: list[FlatQuantileModel] = []

    def factory() -> FlatQuantileModel:
        model = FlatQuantileModel(levels={"0.9": 500.0})
        built.append(model)
        return model

    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON, calibration_days=10, factory=factory
    )
    calibrator.fit(_frame([10.0] * 30))
    assert calibrator.offsets[0.9] == pytest.approx(-490.0)

    built[1].levels["0.9"] = 5.0
    assert (calibrator.predict_quantiles(_frame([0.0] * 5))["0.9"] == 0.0).all()


def test_a_closed_store_is_still_predicted_to_sell_nothing_after_calibration() -> None:
    """An offset must not put stock into a shop that is shut on the day."""
    model = _flat(**{"0.9": 40.0}).fit(_frame([100.0] * 30))
    future = _frame([0.0] * 5, open_flags=[0] * 5)
    assert model.predict_quantiles(future)["0.9"].to_numpy() == pytest.approx([0.0] * 5)


def test_calibrated_quantiles_are_re_sorted_because_offsets_need_not_be_monotone() -> None:
    """Two levels corrected in opposite directions can cross, and a crossed pair is nonsense.

    The 0.5 is corrected up by 90 and the 0.9 down by 900. Move the deployed model so the
    raw pair is already inverted and the two corrections drive them further apart, and
    only the re-sort can put the frame back in order.
    """
    built: list[FlatQuantileModel] = []

    def factory() -> FlatQuantileModel:
        model = FlatQuantileModel(levels={"0.5": 10.0, "0.9": 1000.0})
        built.append(model)
        return model

    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON, calibration_days=10, factory=factory
    )
    calibrator.fit(_frame([100.0] * 30))
    built[1].levels.update({"0.5": 500.0, "0.9": 0.0})

    predicted = calibrator.predict_quantiles(_frame([0.0] * 5))
    assert (predicted["0.5"] <= predicted["0.9"]).all()
    assert predicted["0.9"].to_numpy() == pytest.approx([590.0] * 5)


def test_the_scale_floor_keeps_a_zero_prediction_out_of_the_denominator() -> None:
    """A shut store predicts zero, and an infinity in a pooled quantile is not a number."""
    assert SCALE_FLOOR > 0.0


def test_asking_for_quantiles_before_fitting_is_an_error_rather_than_an_empty_frame() -> None:
    model = _flat(**{"0.9": 40.0})
    with pytest.raises(BacktestError, match="before fit"):
        model.predict_quantiles(_frame([1.0]))
    with pytest.raises(BacktestError, match="only after fit"):
        _ = model.quantiles


def test_a_calibration_window_below_one_day_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="at least 1 day"):
        ConformalQuantileForecaster(horizon=HORIZON, calibration_days=0)


def test_a_horizon_below_one_day_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="horizon must be at least 1 day"):
        ConformalQuantileForecaster(horizon=0)


def test_the_calibration_window_defaults_to_one_horizon() -> None:
    """Residuals gathered a week out say nothing about the spread six weeks out."""
    assert ConformalQuantileForecaster(horizon=HORIZON).calibration_days == HORIZON


def test_the_registry_offers_the_calibrated_model_by_name() -> None:
    assert "gbm_conformal" in FORECASTER_NAMES
    built = forecaster("gbm_conformal", horizon=HORIZON)()
    assert isinstance(built, ConformalQuantileForecaster)
    assert built.horizon == HORIZON


def test_calibration_lifts_out_of_sample_coverage_onto_its_stated_level(
    sales: pd.DataFrame,
) -> None:
    """The claim this module exists to make, against real boosters and held-out data.

    The uncalibrated model covers roughly 0.72 at a nominal 0.9 — pinned in
    `test_gbm.py::test_out_of_sample_the_quantiles_under_cover`. Conformal offsets are
    marginal rather than conditional, and the probe is fitted on less data than the model
    it corrects, so the target is a band around 0.9 and not the number itself.
    """
    fold = rolling_origin(sales[s.DATE], n_folds=1, horizon=HORIZON)[-1]
    train, test = split_frame(sales, fold)

    predicted = ConformalQuantileForecaster(horizon=HORIZON).fit(train).predict_quantiles(test)
    trading = test[s.OPEN] == 1
    covered = coverage(test.loc[trading, s.SALES], predicted.loc[trading, "0.9"])

    assert covered == pytest.approx(0.9, abs=0.1), f"0.9 quantile covered {covered:.3f}"


def test_the_calibrated_median_is_still_a_forecast_and_not_a_stock_level(
    sales: pd.DataFrame,
) -> None:
    """`predict` has to keep meaning what it means, or the backtest harness scores noise."""
    fold = rolling_origin(sales[s.DATE], n_folds=1, horizon=HORIZON)[-1]
    train, test = split_frame(sales, fold)

    model = ConformalQuantileForecaster(horizon=HORIZON).fit(train)
    predicted = model.predict(test)
    trading = test[s.OPEN] == 1

    assert predicted.name is None
    assert model.calibration_rows > 0
    assert coverage(test.loc[trading, s.SALES], predicted[trading]) == pytest.approx(0.5, abs=0.2)
