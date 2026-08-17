"""Assemble the model matrix, and own the list of columns a model may never see.

`FEATURE_DENYLIST` is the second leakage guard. The first (`features.lags`) stops you
building a feature that reaches forward in time. This one stops you using a column that
exists in the training file and simply does not exist at forecast time.

For Rossmann that column is `customers`. It correlates with `sales` at roughly 0.9, it
is in train.csv, and nobody knows it six weeks ahead. A model handed it reports a
wonderful score and cannot be deployed for a single day.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..data import schemas as s
from .calendar import add_calendar
from .lags import add_lags, add_rolling, seasonal_lags

#: Never a feature: the target itself, the date it is indexed by, the raw categorical
#: string replaced by `is_state_holiday`, and anything unknown at the forecast origin.
FEATURE_DENYLIST: frozenset[str] = frozenset(
    {s.DATE, s.STATE_HOLIDAY} | set(s.TARGET_COLUMNS) | set(s.UNAVAILABLE_AT_FORECAST_TIME)
)

DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 28, 91)


def build_features(
    frame: pd.DataFrame,
    *,
    horizon: int,
    lags: Sequence[int] | None = None,
    windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Calendar + lag + rolling features for a horizon-day-ahead forecast.

    `lags` defaults to the first four weekly lags that clear the horizon, because
    same-weekday history is what carries the signal in retail.
    """
    chosen = list(lags) if lags is not None else seasonal_lags(horizon)
    out = add_calendar(frame)
    out = add_lags(out, lags=chosen, horizon=horizon)
    out = add_rolling(out, windows=windows, horizon=horizon)
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column a model is allowed to train on, in a stable order.

    Numeric columns qualify on dtype. Non-numeric ones qualify only by being named in
    the schema as categorical or as the text column — `features/preprocess.py` has a
    branch for each, and a column with no branch has no encoding, so admitting it would
    hand an estimator a string it cannot use.

    This used to return numeric columns alone and drop the rest in silence. That was
    right when every model was a tree fed hand-made flags, and became wrong the moment a
    `ColumnTransformer` existed to encode them: `store_type` and `promo_interval` would
    have been dropped without a word, and the one-hot and Tf-idf branches would have sat
    there with nothing to do.
    """
    encodable = {*s.CATEGORICAL_FEATURES, s.TEXT_FEATURE}
    return [
        column
        for column in frame.columns
        if column not in FEATURE_DENYLIST
        and (pd.api.types.is_numeric_dtype(frame[column]) or column in encodable)
    ]


def assert_no_denied_columns(columns: Sequence[str]) -> None:
    """Raise if a caller has hand-assembled a feature list containing a denied column."""
    denied = sorted(set(columns) & FEATURE_DENYLIST)
    if denied:
        raise ValueError(
            f"{denied} may not be used as features: unknown at the forecast origin, "
            "or the target itself"
        )
