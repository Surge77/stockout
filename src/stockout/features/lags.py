"""Lag and rolling features that cannot see the future.

Three rules, all enforced rather than documented:

**Shift before you roll.** `groupby(store).sales.rolling(7).mean()` includes the current
row. The mean of the last seven days *including today* is not knowable at forecast time,
and a model given it will look excellent and be useless.

**A lag must be at least the horizon.** Predicting six weeks out, you do not know
yesterday's sales — you are standing at the forecast origin, six weeks back. `lag_1` in a
horizon-42 model is the most common leak in public retail notebooks and the hardest to
spot, because nothing about the code looks wrong. Here it raises.

**Group before you shift.** A shift across a concatenated frame pulls the previous
store's last day into the next store's first. Every operation is grouped.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..data import schemas as s
from ..errors import LeakageError


def add_lags(
    frame: pd.DataFrame,
    *,
    column: str = s.SALES,
    lags: Sequence[int],
    horizon: int,
    group: str = s.STORE,
) -> pd.DataFrame:
    """Append `{column}_lag_{n}` for each n in `lags`.

    Raises `LeakageError` if any lag is shorter than `horizon`, because such a feature
    is unobservable at the forecast origin.
    """
    _reject_short_lags(lags, horizon)
    out = frame.copy()
    grouped = out.groupby(group, observed=True)[column]
    for lag in lags:
        out[f"{column}_lag_{lag}"] = grouped.shift(lag)
    return out


def add_rolling(
    frame: pd.DataFrame,
    *,
    column: str = s.SALES,
    windows: Sequence[int],
    horizon: int,
    group: str = s.STORE,
) -> pd.DataFrame:
    """Append `{column}_roll_mean_{w}` and `{column}_roll_std_{w}` for each window.

    Each window ends `horizon` days before the target date: the frame is shifted by
    `horizon` first, then rolled. A window computed without that shift would include
    the target itself.
    """
    if any(w < 1 for w in windows):
        raise ValueError("rolling windows must be positive")

    out = frame.copy()
    shifted = out.groupby(group, observed=True)[column].shift(horizon)
    by_store = shifted.groupby(out[group], observed=True)
    for window in windows:
        rolled = by_store.rolling(window, min_periods=1)
        out[f"{column}_roll_mean_{window}"] = rolled.mean().reset_index(level=0, drop=True)
        out[f"{column}_roll_std_{window}"] = rolled.std().reset_index(level=0, drop=True)
    return out


def _reject_short_lags(lags: Sequence[int], horizon: int) -> None:
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    if any(lag < 1 for lag in lags):
        raise ValueError("lags must be positive")

    too_short = sorted(lag for lag in lags if lag < horizon)
    if too_short:
        raise LeakageError(
            f"lags {too_short} are shorter than the {horizon}-day horizon. At the forecast "
            f"origin those values have not happened yet. Use lags >= {horizon}."
        )


def seasonal_lags(horizon: int, *, season: int = s.SUNDAY, count: int = 4) -> list[int]:
    """The first `count` multiples of `season` that clear `horizon`.

    Same-weekday history is the useful signal in retail, so lags should land on
    multiples of seven rather than on arbitrary round numbers.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    first = ((horizon + season - 1) // season) * season
    return [first + season * i for i in range(count)]
