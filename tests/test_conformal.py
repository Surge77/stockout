"""Calibration: the modelling against a stub, then the claim against real boosters.

The arithmetic these tests used to cover now lives in `test_conformity.py`. What is left
here is the modelling — whose residuals, measured on which window, and what happens to a
shut store — and it is checked against `FlatQuantileModel`, which returns predictions with
a known answer already in them, so a wrong offset is a failed assertion rather than a
slightly different third decimal place.

The two tests at the bottom fit the real thing, because a calibration layer that works on
a stub and not on a booster has calibrated nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.errors import BacktestError
from stockout.evaluate.metrics import coverage
from stockout.models import FORECASTER_NAMES, forecaster
from stockout.models.conformal import ConformalQuantileForecaster
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


# --- per-group calibration (ADR 0012) -------------------------------------------------


def _two_store_frame() -> pd.DataFrame:
    """Two stores over thirty days. Store 1 sells 100, store 2 sells 300."""
    days = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            s.DATE: list(days) * 2,
            s.STORE: [1] * 30 + [2] * 30,
            s.SALES: [100.0] * 30 + [300.0] * 30,
            s.OPEN: [1] * 60,
        }
    ).sort_values([s.STORE, s.DATE], ignore_index=True)


def _grouped(**levels: float) -> ConformalQuantileForecaster:
    return ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=10,
        factory=lambda: FlatQuantileModel(levels=levels),
        group_by=s.STORE,
        min_group_rows=3,
    )


def test_grouping_is_off_unless_it_is_asked_for() -> None:
    """The marginal correction stays the default, so no published number moves."""
    model = _flat(**{"0.5": 40.0}).fit(_frame([100.0] * 30))
    assert model.group_offsets == {}
    assert model.pooled_fallback_groups == ()


def test_each_store_is_corrected_by_what_that_store_did() -> None:
    """Mondrian, hand-checkable. One offset would fit neither of these two stores.

    Both stores are predicted flat at 40 with a scale of 1. Store 1 is short by 60 on
    every calibration day and store 2 by 260, so a pooled offset lands between them and
    is wrong for both. Per store, each gets its own.
    """
    model = _grouped(**{"0.5": 40.0}).fit(_two_store_frame())

    assert model.pooled_fallback_groups == ()
    assert model.group_offsets[1][0.5] == pytest.approx(60.0)
    assert model.group_offsets[2][0.5] == pytest.approx(260.0)


def test_a_grouped_prediction_shifts_each_store_by_its_own_offset() -> None:
    """The offsets have to reach the predictions, not merely be computed and stored."""
    model = _grouped(**{"0.5": 40.0}).fit(_two_store_frame())

    future = pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-03-01", periods=2, freq="D").tolist() * 2,
            s.STORE: [1, 1, 2, 2],
            s.SALES: [0.0] * 4,
            s.OPEN: [1] * 4,
        }
    )
    predicted = model.predict_quantiles(future)["0.5"].to_numpy()
    assert predicted == pytest.approx([100.0, 100.0, 300.0, 300.0])


def test_a_store_too_thin_to_calibrate_takes_the_marginal_offset_and_is_named() -> None:
    """Below the floor the offsets fit noise, which ADR 0009 measured at 70 pooled rows."""
    model = ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=10,
        factory=lambda: FlatQuantileModel(levels={"0.5": 40.0}),
        group_by=s.STORE,
        min_group_rows=99,
    ).fit(_two_store_frame())

    assert model.pooled_fallback_groups == (1, 2)
    assert model.group_offsets == {}


def test_the_row_floor_defaults_to_what_the_strictest_level_needs() -> None:
    """Derived from the grid, so asking for a 0.99 per store is expensive by arithmetic."""
    model = ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=10,
        factory=lambda: FlatQuantileModel(levels={"0.5": 40.0, "0.99": 40.0}),
        group_by=s.STORE,
    ).fit(_two_store_frame())

    assert model.min_group_rows == 199
    assert model.pooled_fallback_groups == (1, 2)


def test_a_store_the_calibration_window_never_saw_is_named_rather_than_silently_pooled() -> None:
    """A shop that opened last week gets the marginal correction, and it is not a secret."""
    model = _grouped(**{"0.5": 40.0}).fit(_two_store_frame())

    future = pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-03-01", periods=2, freq="D"),
            s.STORE: [1, 99],
            s.SALES: [0.0, 0.0],
            s.OPEN: [1, 1],
        }
    )
    predicted = model.predict_quantiles(future)["0.5"].to_numpy()

    assert model.unseen_groups == (99,)
    # Store 1 takes its own +60; store 99 takes the pooled offset over both stores.
    assert predicted[0] == pytest.approx(100.0)
    assert predicted[1] != pytest.approx(100.0)


def test_calibrating_by_a_column_that_does_not_exist_is_refused_at_fit() -> None:
    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=10,
        factory=lambda: FlatQuantileModel(levels={"0.5": 40.0}),
        group_by="region",
    )
    with pytest.raises(BacktestError, match="no such column"):
        calibrator.fit(_two_store_frame())


def test_predicting_a_frame_without_the_grouping_column_is_refused() -> None:
    """Silently falling back to marginal here would make the coverage claim untrue quietly."""
    model = _grouped(**{"0.5": 40.0}).fit(_two_store_frame())
    future = pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-03-01", periods=2, freq="D"),
            s.SALES: [0.0, 0.0],
            s.OPEN: [1, 1],
        }
    )
    with pytest.raises(BacktestError, match="no such column"):
        model.predict_quantiles(future)
