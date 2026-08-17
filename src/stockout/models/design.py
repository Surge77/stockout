"""The design matrix a booster is fitted on, and the one it predicts against.

Split from `gbm.py` because the two are different jobs. This module answers "which rows
and which columns, built from what history"; `gbm.py` answers "what objective, how many
rounds". The feature plumbing is also the half where the leakage guards live, and it is
worth reading without a boosting loop wrapped around it.

**Why `fit` keeps the training frame.** Every lag here is at least one horizon long, so
the features for a 42-day test window resolve entirely into training history — that is
the whole point of the leakage guard, and it means `predict` cannot build a feature row
from the future frame alone. It needs the history that sits behind it.

**Why the training rows and the retained history are separate arguments.** They are the
same frame in the ordinary case and must not be in one specific case: conformal
calibration wants a model that has never *trained* on the last 42 days but can still
*build features* across them, because a test row's `lag_42` lands inside exactly those
days. Retaining only the inner window would leave a hole in the calendar, and lags are
built by position, so a shift of 42 rows across a 42-day gap silently lands 84 days back.
ADR 0013 turns on that distinction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..data import schemas as s
from ..errors import BacktestError
from ..features.build import build_features, feature_columns
from .base import open_rows, zero_when_closed

if TYPE_CHECKING:
    import lightgbm as lgb

#: `seed` here is a seeded *sampler* — row and column subsampling inside boosting — and
#: not a seeded *split*. The distinction is the one `tests/test_no_random_splits.py`
#: exists to police: randomness that changes which rows a tree sees is fine, randomness
#: that decides which rows are training and which are testing is not.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
}


class GbmDesign:
    """Feature plumbing shared by the point and quantile forecasters."""

    name = "gbm_base"

    def __init__(self, *, horizon: int, params: dict[str, object] | None = None) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1 day")
        self.horizon = horizon
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS if params is None else params)
        self._history = pd.DataFrame()
        self._features: list[str] = []

    @property
    def features(self) -> list[str]:
        """The columns the model was fitted on, in the order it expects them."""
        return list(self._features)

    def _training_matrix(
        self, train: pd.DataFrame, *, history: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Design matrix and target, restricted to trading days with defined lags.

        Closed days are dropped for the same reason the baselines drop them: a zero on a
        shut Sunday is not a demand observation, and averaging it in drags every level
        estimate down by roughly a seventh. `zero_when_closed` puts the zeros back at
        prediction time, where they belong.

        `history` defaults to `train` and is the frame features are built from and
        retained for prediction. Passing a wider one boosts on `train`'s rows while
        keeping the whole calendar available to lag against — see the module docstring.
        """
        source = train if history is None else history
        self._history = source.copy()
        matrix = build_features(source, horizon=self.horizon)
        if history is not None:
            matrix = matrix.loc[_rows_in(matrix, train)]

        usable = open_rows(matrix).dropna(subset=_lag_columns(matrix))
        if usable.empty:
            raise BacktestError(
                f"no training row has a defined lag at horizon {self.horizon}; the "
                "training window is shorter than the lags it implies"
            )
        self._features = feature_columns(usable)
        return usable[self._features], usable[s.SALES]

    def _future_matrix(self, future: pd.DataFrame) -> pd.DataFrame:
        """Feature rows for `future`, built against the retained training history.

        Rows already present in history keep their observed `sales` so that *later*
        rows can lag them. Rows the model has not seen get a NaN target: the protocol
        says a future frame's `sales` column must be treated as absent, and it is only
        there because slicing a frame keeps every column.

        Private, and reached only after a caller has checked that a fit happened.
        """
        keys = pd.MultiIndex.from_frame(future[list(s.KEY_COLUMNS)])
        known = pd.MultiIndex.from_frame(self._history[list(s.KEY_COLUMNS)])

        unseen = future.loc[~np.asarray(keys.isin(known))].reindex(
            columns=self._history.columns
        )
        unseen = unseen.assign(**{s.SALES: np.nan})

        combined = pd.concat([self._history, unseen], ignore_index=True)
        combined = combined.sort_values(list(s.KEY_COLUMNS)).reset_index(drop=True)

        matrix = build_features(combined, horizon=self.horizon)
        # Located rather than reindexed: `store` is itself a feature, so the key columns
        # have to stay in the frame instead of moving into the index. Every key resolves
        # by construction — `combined` is the history plus exactly the rows that were
        # missing from it.
        positions = pd.MultiIndex.from_frame(matrix[list(s.KEY_COLUMNS)]).get_indexer(keys)
        return matrix.iloc[positions][self._features]

    def _predict_with(self, booster: lgb.Booster, future: pd.DataFrame) -> pd.Series:
        raw = np.asarray(booster.predict(self._future_matrix(future)), dtype="float64")
        values = pd.Series(raw, index=future.index).clip(lower=0.0)
        return zero_when_closed(values, future)


def _rows_in(matrix: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """Boolean mask of `matrix` rows whose store and date appear in `train`."""
    wanted = pd.MultiIndex.from_frame(train[list(s.KEY_COLUMNS)])
    present = pd.MultiIndex.from_frame(matrix[list(s.KEY_COLUMNS)])
    return np.asarray(present.isin(wanted))


def _lag_columns(matrix: pd.DataFrame) -> list[str]:
    prefix = f"{s.SALES}_lag_"
    return [column for column in matrix.columns if column.startswith(prefix)]
