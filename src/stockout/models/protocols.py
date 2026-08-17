"""What the calibration layer requires of the model it wraps.

Two contracts, because two different things are being asked for and only one of them is
cheap. Kept out of `conformal.py` so that "what a wrapped model must provide" is readable
without the calibration arithmetic around it, and so that a stub in a test can be checked
against the contract rather than against a class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


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


@runtime_checkable
class HistoryAwareQuantileModel(QuantileModel, Protocol):
    """A model that can be trained on some rows while lagging against more of them.

    Required only by the no-refit calibration of ADR 0013, and separate from
    `QuantileModel` because most models have no reason to offer it: the ability to hold
    back the last 42 days from *training* while still using them to *build features* is a
    peculiar thing to need, and it is needed for exactly one reason. Runtime-checkable so
    that asking for the no-refit mode with a model that cannot serve it fails with a
    sentence rather than an AttributeError.
    """

    def fit_within(
        self, train: pd.DataFrame, *, history: pd.DataFrame
    ) -> HistoryAwareQuantileModel: ...
