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
misalign them. By default a probe model is fitted on the inner window and scored on the
calibration window, and the deployed model is fitted on all of it.

That refit is what costs the guarantee. The residuals describe the probe, the predictions
come from the deployed model, and the two are not exchangeable, so the coverage statement
becomes an expectation rather than a theorem. The direction of the error is at least
arguable: the probe has seen less data, so its residuals are no smaller than the deployed
model's and the offsets it yields should if anything over-correct. That is an argument, not
a proof, and the measured coverage is what the default claim rests on.

**`refit=False` serves the probe, and buys back exactly one of two missing premises.** The
lag alignment that made this look impossible was never a *training* problem, it was a
*feature-history* problem: `design.GbmDesign` can boost on the inner window while
retaining the whole training frame to lag against, so the deployed model keeps a complete
calendar and has still never trained on the calibration window. The offsets then describe
the estimator that serves them, and the split-conformal theorem applies to it.

It does not follow that coverage is proven, and this module will not say that it is. The
theorem needs the calibration and test rows to be exchangeable, and time-ordered retail
demand is not: the test window comes after the calibration window, its distribution has
moved, and same-day rows across stores are correlated. What changed is that the refit is
no longer *also* in the way — one assumption remains instead of two, and it is named
rather than absorbed. ADR 0013 measures what the swap costs in accuracy.

Fitting costs twice as long with the refit and once without it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from ..data import schemas as s
from ..errors import BacktestError
from .base import zero_when_closed
from .conformity import (
    floored_scale,
    grouped_offsets,
    groups_without_offsets,
    min_rows_for,
    offsets_for_rows,
    pooled_offsets,
    saturates,
    split_calibration_window,
)
from .gbm import DEFAULT_NUM_BOOST_ROUND, DEFAULT_QUANTILES, GbmQuantileForecaster
from .protocols import HistoryAwareQuantileModel, QuantileModel

__all__ = ["ConformalQuantileForecaster", "QuantileModel"]


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
        group_by: str | None = None,
        min_group_rows: int | None = None,
        refit: bool = True,
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

        # Mondrian calibration, off by default. `group_offsets` stays empty unless asked
        # for, `pooled_fallback_groups` names the groups too thin to estimate, and both are
        # public for the same reason `offsets` is: a correction nobody can inspect is a
        # correction nobody can defend.
        self.group_by = group_by
        self.min_group_rows = min_group_rows
        self.group_offsets: dict[object, dict[float, float]] = {}
        self.pooled_fallback_groups: tuple[object, ...] = ()
        self.unseen_groups: tuple[object, ...] = ()

        # True keeps the two-fit design of ADR 0009: the deployed model is refitted on the
        # whole window and the coverage claim is measured. False serves the probe itself, so
        # the offsets describe the estimator that produced them and the split-conformal
        # theorem applies to it — at the price of the most recent horizon of training data.
        # ADR 0013, which is also careful about what that does and does not prove.
        self.refit = refit

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

        probe = self._probe(inner, train)
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
        if self.group_by is not None:
            self._fit_groups(predicted=predicted, probe=probe, calibration=calibration)

        # The refit is what costs the theorem, so not refitting is what recovers it: the
        # residuals above were measured on this very estimator rather than on a discarded
        # one. It is also the more expensive option in accuracy, because the model that
        # goes out has never seen the most recent horizon of history.
        self._model = self._factory().fit(train) if self.refit else probe
        return self

    def _probe(self, inner: pd.DataFrame, train: pd.DataFrame) -> QuantileModel:
        """The model whose residuals become the offsets.

        Trained on `inner` either way. When it is going to be *deployed* it also needs the
        whole of `train` to lag against, because a test row's lag lands inside the
        calibration window and a model retaining only the inner window would shift across
        that hole by position and read the wrong dates. That distinction is the whole
        reason ADR 0009 refitted, and `fit_within` is what removes the need to.
        """
        if self.refit:
            return self._factory().fit(inner)

        built = self._factory()
        if not isinstance(built, HistoryAwareQuantileModel):
            raise BacktestError(
                f"refit=False needs a model offering fit_within(train, history=...) so the "
                f"calibration window can be held out of training without being held out of "
                f"the feature history; {type(built).__name__} does not"
            )
        return built.fit_within(inner, history=train)

    def _fit_groups(
        self, *, predicted: pd.DataFrame, probe: QuantileModel, calibration: pd.DataFrame
    ) -> None:
        """One offset set per group, for the groups whose row count can carry one."""
        if self.group_by not in calibration.columns:
            raise BacktestError(
                f"cannot calibrate by {self.group_by!r}: the training frame has no such "
                f"column. Available: {', '.join(map(str, calibration.columns))}"
            )

        floor = (
            min_rows_for(float(column) for column in predicted.columns)
            if self.min_group_rows is None
            else self.min_group_rows
        )
        self.min_group_rows = floor
        self.group_offsets, self.pooled_fallback_groups = grouped_offsets(
            predicted=predicted,
            scale=probe.predict(calibration),
            frame=calibration,
            groups=calibration[self.group_by],
            min_rows=floor,
        )

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
                (raw[column] + self._offset_for(str(column), future) * scale).clip(lower=0.0),
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

    def _offset_for(self, column: str, future: pd.DataFrame) -> pd.Series | float:
        """The correction for one level: a scalar when pooled, a column when grouped.

        A group the calibration window never showed — a store that opened since, or a
        group too thin to estimate — takes the marginal offset. That is the right
        fallback and a silent one, so the names are kept on `unseen_groups` and
        `pooled_fallback_groups`; degrading from a conditional correction to a marginal
        one without saying so is how a coverage claim stops being true quietly.
        """
        level = float(column)
        pooled = self.offsets.get(level, 0.0)
        key = self.group_by
        if key is None or not self.group_offsets:
            return pooled

        if key not in future.columns:
            raise BacktestError(
                f"calibrated by {key!r} but the frame being predicted has no such column"
            )
        labels = future[key]
        self.unseen_groups = groups_without_offsets(labels, self.group_offsets)
        return offsets_for_rows(
            labels=labels, group_offsets=self.group_offsets, pooled=pooled, level=level
        )

    def _split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return split_calibration_window(train, days=self.calibration_days)
