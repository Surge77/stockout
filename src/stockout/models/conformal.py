"""Conformal-style calibration for a quantile forecaster.

**Why this module exists.** `GbmQuantileForecaster` minimises pinball loss and is honest
about it, but out of sample its nominal 0.9 covers roughly 0.72 of trading days —
`tests/test_gbm.py::test_out_of_sample_the_quantiles_under_cover` pins that defect on
purpose. A service level that does not deliver its own number is not a service level, and
every cost the frontier prices downstream is then the cost of a policy nobody chose. The
pinball fit is not wrong; it is asked to bracket a spread six weeks ahead that it only
ever saw one week ahead, and a tree cannot extrapolate variance it was never shown.

**What is done about it.** One additive offset per quantile, learned on a slice of
training data the boosters never saw, in the geometry they will meet at deployment. It
fits no distribution: no normality, no variance model, nothing taken from the residuals
but an order statistic of them.

The arithmetic of that order statistic — the finite-sample correction, the saturation
test and the scale floor — lives in `conformity.py`, which has no model in it and can be
checked against numbers written by hand. What is left here is the modelling: whose
residuals, measured on which window, and what happens to a shut store.

**What it is not, and this matters.** Textbook split conformal proves marginal coverage
for *the model that produced the residuals*. The model deployed here is not that model —
see "why two fits" below — so the proof does not transfer and nothing in this module
carries it. The claim is narrower and is measured rather than proven: the correction moves
out-of-sample coverage substantially towards the stated level, `stockout calibration`
prints the number, and ADR 0009 records the draw where it moves the wrong way.

**Why the residual is scaled rather than raw.** A pooled offset in currency units
over-corrects a quiet store and under-corrects a busy one; on the committed sample the
store means differ by a factor of two, and on Rossmann by far more. Retail error variance
scales with level — `gbm.py` says so in its own docstring — so the conformity score is
divided by the model's median prediction for that row before it is pooled. The offset is
then a relative correction and travels across stores.

**Why two fits, and what it costs.** Split conformal wants the offsets measured on the
model that will be deployed. The deployed model wants every day of history, and its lag
features are built by *position*, so a hole punched in its calendar would silently
misalign them. Rather than trade one for the other, a probe model is fitted on the inner
window and scored on the calibration window, and the deployed model is fitted on all of
it.

This is the refit that costs the guarantee. The residuals describe the probe, the
predictions come from the deployed model, and the two are not exchangeable, so the
coverage statement becomes an expectation rather than a theorem. The direction of the
error is at least arguable: the probe has seen less data, so its residuals are no smaller
than the deployed model's and the offsets it yields should if anything over-correct. That
is an argument, not a proof, and the measured coverage is what the claim rests on.

Serving predictions from the probe instead would restore the theorem and break the lag
alignment it exists to protect, which is a worse trade. Fitting costs twice as long either
way.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from ..data import schemas as s
from ..errors import BacktestError
from .base import zero_when_closed
from .conformity import floored_scale, pooled_offsets, saturates
from .gbm import DEFAULT_NUM_BOOST_ROUND, DEFAULT_QUANTILES, GbmQuantileForecaster


class QuantileModel(Protocol):
    """What calibration needs from the model it wraps.

    Narrow on purpose. It exists so the calibration arithmetic can be tested against a
    two-line stub in milliseconds instead of against six boosters, and because a
    calibration layer that only works on LightGBM is a calibration layer nobody can
    check.
    """

    crossing_rate: float

    @property
    def quantiles(self) -> tuple[float, ...]: ...

    def fit(self, train: pd.DataFrame) -> QuantileModel: ...

    def predict(self, future: pd.DataFrame) -> pd.Series: ...

    def predict_quantiles(self, future: pd.DataFrame) -> pd.DataFrame: ...


class ConformalQuantileForecaster:
    """A quantile forecaster whose stated levels are made to mean what they say.

    `offsets` and `calibration_rows` are public because a calibration you cannot inspect
    is a calibration you cannot defend. An offset of +0.31 says the raw 0.9 quantile sat
    a third of a median below where it needed to be, and that is a fact about the model
    worth reading rather than a constant to be applied quietly.
    """

    name = "gbm_conformal"

    def __init__(
        self,
        *,
        horizon: int,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        params: dict[str, object] | None = None,
        num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
        calibration_days: int | None = None,
        factory: Callable[[], QuantileModel] | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1 day")
        # One horizon by default, because the calibration window has to be entered from
        # the same distance the test window is. Residuals gathered one week ahead say
        # nothing about the spread six weeks ahead, which is the whole defect being fixed.
        self.calibration_days = horizon if calibration_days is None else calibration_days
        if self.calibration_days < 1:
            raise ValueError("the calibration window must be at least 1 day")

        self.horizon = horizon
        self.offsets: dict[float, float] = {}
        self.saturated_quantiles: tuple[float, ...] = ()
        self.calibration_rows = 0
        self.crossing_rate: float = float("nan")

        self._factory: Callable[[], QuantileModel] = factory or (
            lambda: GbmQuantileForecaster(
                horizon=horizon,
                quantiles=quantiles,
                params=params,
                num_boost_round=num_boost_round,
            )
        )
        self._model: QuantileModel | None = None

    @property
    def quantiles(self) -> tuple[float, ...]:
        if self._model is None:
            raise BacktestError("quantiles are known only after fit()")
        return tuple(self._model.quantiles)

    def fit(self, train: pd.DataFrame) -> ConformalQuantileForecaster:
        """Learn the offsets on a held-out tail, then fit the model that will be used."""
        inner, calibration = self._split(train)

        probe = self._factory().fit(inner)
        predicted = probe.predict_quantiles(calibration)
        self.offsets = pooled_offsets(
            predicted=predicted,
            scale=probe.predict(calibration),
            frame=calibration,
        )
        self.calibration_rows = int((calibration[s.OPEN] == 1).sum())
        self.saturated_quantiles = tuple(
            level for level in sorted(self.offsets) if saturates(self.calibration_rows, level)
        )

        self._model = self._factory().fit(train)
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """The calibrated median, so this satisfies the same contract as every forecaster."""
        predicted = self.predict_quantiles(future)
        median = min(self.quantiles, key=lambda q: abs(q - 0.5))
        return predicted[str(median)].rename(None)

    def predict_quantiles(self, future: pd.DataFrame) -> pd.DataFrame:
        """Raw quantiles shifted by their offsets, re-sorted, and zeroed on closures."""
        if self._model is None:
            raise BacktestError("predict_quantiles() was called before fit()")

        raw = self._model.predict_quantiles(future)
        self.crossing_rate = self._model.crossing_rate
        scale = floored_scale(self._model.predict(future))

        shifted = {
            column: zero_when_closed(
                (raw[column] + self.offsets.get(float(column), 0.0) * scale).clip(lower=0.0),
                future,
            )
            for column in raw.columns
        }
        adjusted = pd.DataFrame(shifted, index=raw.index, columns=raw.columns)
        # Re-sorted because the offsets are learned per quantile and nothing forces them
        # to be monotone; a 0.8 that overtakes a 0.9 would price a nonsense policy.
        return pd.DataFrame(
            np.sort(adjusted.to_numpy(dtype="float64"), axis=1),
            index=adjusted.index,
            columns=adjusted.columns,
        )

    def _split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inner training window and calibration tail, divided by date and never by row.

        Dividing by row would put some of a day's stores on one side and the rest on the
        other, which is a random split wearing a timestamp.
        """
        dates = pd.to_datetime(train[s.DATE])
        cutoff = pd.Timestamp(dates.max()) - pd.Timedelta(days=self.calibration_days)
        inner = train[dates <= cutoff]
        calibration = train[dates > cutoff]

        if inner.empty:
            raise BacktestError(
                f"a {self.calibration_days}-day calibration window leaves no training "
                f"data; the training frame spans {len(dates.unique())} days"
            )
        if not bool((calibration[s.OPEN] == 1).any()):
            raise BacktestError(
                "the calibration window contains no trading day, so no residual can be "
                "measured; lengthen it with calibration_days"
            )
        return inner, calibration
