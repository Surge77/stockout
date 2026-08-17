"""Calibration in its two optional modes: grouped (ADR 0012) and un-refitted (ADR 0013).

Split from `test_conformal.py`, which covers the marginal correction that is on by
default. Both modes here are off unless asked for, and both change *which model or which
rows* the offsets come from rather than the arithmetic that turns residuals into offsets —
that is `test_conformity.py`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import FlatQuantileModel, flat_calibrator, one_store_frame

from stockout.data import schemas as s
from stockout.errors import BacktestError
from stockout.evaluate.metrics import coverage
from stockout.models.conformal import ConformalQuantileForecaster
from stockout.split.rolling import rolling_origin, split_frame

HORIZON = 42




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
    model = flat_calibrator(**{"0.5": 40.0}).fit(one_store_frame([100.0] * 30))
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


# --- the no-refit mode (ADR 0013) -----------------------------------------------------


class HistoryAwareFlatModel(FlatQuantileModel):
    """A flat model that records what it was trained on and what it may lag against.

    The distinction the no-refit mode turns on is invisible in a prediction and visible
    here: `fitted_rows` counts the boosting rows and `history_rows` the calendar the
    features were built from. A correct implementation trains on fewer rows than it
    retains.
    """

    def __init__(self, *, levels: dict[str, float]) -> None:
        super().__init__(levels=levels)
        self.history_rows = 0

    def fit_within(
        self, train: pd.DataFrame, *, history: pd.DataFrame
    ) -> HistoryAwareFlatModel:
        self.fitted_rows = len(train)
        self.history_rows = len(history)
        return self


def test_by_default_the_deployed_model_is_a_second_fit() -> None:
    """ADR 0009's design, unchanged, because every published number depends on it."""
    built: list[FlatQuantileModel] = []

    def factory() -> FlatQuantileModel:
        model = FlatQuantileModel(levels={"0.9": 40.0})
        built.append(model)
        return model

    ConformalQuantileForecaster(
        horizon=HORIZON, calibration_days=10, factory=factory
    ).fit(one_store_frame([100.0] * 30))

    assert len(built) == 2


def test_without_the_refit_one_model_is_fitted_and_it_is_the_one_that_serves() -> None:
    """The point of ADR 0013: the offsets describe the estimator that produced them."""
    built: list[HistoryAwareFlatModel] = []

    def factory() -> HistoryAwareFlatModel:
        model = HistoryAwareFlatModel(levels={"0.9": 40.0})
        built.append(model)
        return model

    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON, calibration_days=10, factory=factory, refit=False
    ).fit(one_store_frame([100.0] * 30))

    assert len(built) == 1
    assert calibrator.offsets[0.9] == pytest.approx(60.0)


def test_the_served_model_trains_on_the_inner_window_and_lags_across_all_of_it() -> None:
    """The two halves that make the theorem and the lag alignment compatible.

    Twenty inner rows are trained on; all thirty stay available to build features from.
    Holding the calibration window out of the *history* as well is what ADR 0009 said
    would misalign the lags, and it is not what happens here.
    """
    built: list[HistoryAwareFlatModel] = []

    def factory() -> HistoryAwareFlatModel:
        model = HistoryAwareFlatModel(levels={"0.9": 40.0})
        built.append(model)
        return model

    ConformalQuantileForecaster(
        horizon=HORIZON, calibration_days=10, factory=factory, refit=False
    ).fit(one_store_frame([100.0] * 30))

    assert built[0].fitted_rows == 20
    assert built[0].history_rows == 30


def test_asking_for_no_refit_from_a_model_that_cannot_do_it_is_a_sentence_not_a_crash() -> None:
    """`FlatQuantileModel` has no `fit_within`, and the message has to say what is missing."""
    calibrator = ConformalQuantileForecaster(
        horizon=HORIZON,
        calibration_days=10,
        factory=lambda: FlatQuantileModel(levels={"0.9": 40.0}),
        refit=False,
    )
    with pytest.raises(BacktestError, match="needs a model offering fit_within"):
        calibrator.fit(one_store_frame([100.0] * 30))


def test_the_no_refit_model_still_covers_its_level_on_real_boosters(
    sales: pd.DataFrame,
) -> None:
    """A theorem that applies to an estimator nobody can fit is worth nothing.

    The band is the same one the refitting model is held to. What ADR 0013 claims is that
    the offsets now describe the deployed estimator, not that the coverage number improves
    — and the accuracy it gives up is recorded in the ADR rather than asserted here.
    """
    fold = rolling_origin(sales[s.DATE], n_folds=1, horizon=HORIZON)[-1]
    train, test = split_frame(sales, fold)

    model = ConformalQuantileForecaster(horizon=HORIZON, refit=False).fit(train)
    predicted = model.predict_quantiles(test)
    trading = test[s.OPEN] == 1
    covered = coverage(test.loc[trading, s.SALES], predicted.loc[trading, "0.9"])

    assert covered == pytest.approx(0.9, abs=0.15), f"0.9 quantile covered {covered:.3f}"
