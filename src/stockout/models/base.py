"""The forecaster contract, plus the one rule every forecaster obeys.

`zero_when_closed` deserves a note. Whether a store is trading on a given future date is
*known in advance* — it comes from a trading calendar, not from an observation — so using
it is not leakage, and any forecaster that predicts revenue for a shuttered store is
simply wrong rather than optimistic. Closed days are excluded from the metrics anyway,
so this changes no score; it exists so that predictions written to a file are usable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..data import schemas as s


@runtime_checkable
class Forecaster(Protocol):
    """Fit on a training frame, predict for a future frame of the same schema.

    `predict` receives rows whose `sales` column must be treated as absent — it is
    present only because slicing a frame keeps all its columns.
    """

    name: str

    def fit(self, train: pd.DataFrame) -> Forecaster: ...

    def predict(self, future: pd.DataFrame) -> pd.Series: ...


def zero_when_closed(predictions: pd.Series, future: pd.DataFrame) -> pd.Series:
    """Force predictions to zero wherever the trading calendar says the store is shut."""
    if s.OPEN not in future.columns:
        return predictions
    return predictions.where(future[s.OPEN].to_numpy() == 1, 0.0)


def open_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Trading days only.

    Baselines fit on these because a closed day's zero is not a demand observation.
    Averaging it in drags every level estimate down by roughly a seventh.
    """
    if s.OPEN not in frame.columns:
        return frame
    return frame[frame[s.OPEN] == 1]
