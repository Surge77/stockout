"""The part of a store's sales that a tree can never predict: where the level is going.

Gradient boosting and random forests **cannot extrapolate**. A tree splits on a threshold
and returns a leaf value; asked about a point past every threshold it ever built, it lands
in the same terminal leaf as the largest value it saw and answers with that. Handing such a
model a time index does not fix this, which is the trap here — the feature looks like it
should work and buys nothing at all outside the training range.

That is why, before this module, `06 March 2026` and `06 March 2015` returned an identical
number. Not a missing feature: a model class that is structurally unable to produce a value
it never saw.

Rossmann is not flat. Measured over 2013-01-01 to 2015-07-31, the chain grows about 5.1% a
year, the median store 3.7%, and 68% of stores grow faster than 2% while 10% shrink faster
than 2%. A model blind to that is systematically about 5% low one year out and 10% low two
years out, and the p10-to-p90 spread of fourteen points says one chain-wide number will not
do — the slopes are genuinely per store.

So the level goes somewhere a tree is not. A straight line per store, fitted here,
subtracted before the tree sees the target and added back after it answers:

    sales  ≈  (aᵢ + bᵢ·t)  +  tree(calendar, promotions, store attributes)
              └── extrapolates ──┘   └────── does not need to ──────┘

**Fitted on the training rows alone**, exactly like `targets.fit`. A slope is a statistic,
and one measured over the whole frame has let the test period set the line it is scored
against. Per store with a global fallback below `MIN_ROWS_PER_ENTITY`, which is the same
shape `targets.py` already uses for demand terciles.

This is not a lag and nothing here shifts the sales column, so the leakage argument in
`features.py` survives untouched: `t` for any future day is knowable today.

**What a straight line cannot do**, stated rather than discovered later. It assumes the
trend continues — a competitor opening, a refurbishment or a recession breaks it. It cannot
represent a trend that changes. And extrapolated far enough it becomes arithmetic rather
than a forecast, which is what `MAX_CONFIDENT_DAYS` exists to say out loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import schemas as s
from .errors import StockoutError

#: Rows a store needs before its own slope is believed. Below it the chain-wide line is
#: used instead. A slope fitted on a handful of days is noise with a direction, and it is
#: extrapolated forward — the one place a bad estimate compounds instead of averaging out.
MIN_ROWS_PER_ENTITY = 180

#: How far past the data a straight line is worth quoting. Beyond this the arithmetic still
#: works and the answer stops being a forecast: a 5%-a-year line says nothing useful about
#: 2035. Surfaces as a warning rather than a refusal, because where exactly the line stops
#: being meaningful is a judgement the caller is better placed to make than this module.
MAX_CONFIDENT_DAYS = 550


@dataclass(frozen=True)
class StoreTrends:
    """A straight line per store, plus a chain-wide one for stores with too few rows."""

    per_entity: dict[str, tuple[float, float]]
    """entity -> (intercept at `origin`, slope per day)."""

    global_line: tuple[float, float]
    origin: pd.Timestamp
    """The date `t = 0` refers to. Travels because a slope is meaningless without it."""

    fell_back: frozenset[str] = field(default_factory=frozenset)

    def line_for(self, entity: str) -> tuple[float, float]:
        return self.per_entity.get(str(entity), self.global_line)

    def level_at(self, entity: str, date: pd.Timestamp) -> float:
        """Where this store's line sits on one day."""
        intercept, slope = self.line_for(entity)
        return float(intercept + slope * _days_since(date, self.origin))

    def annual_growth(self, entity: str) -> float:
        """The store's slope as a fraction of its own level, which is the readable form.

        A store taking 9,000 a day and gaining 400 a year is growing; one taking 900 and
        gaining the same 400 is a different story entirely, and the raw slope cannot tell
        them apart.
        """
        intercept, slope = self.line_for(entity)
        return float(slope * 365.25 / intercept) if intercept else 0.0

    def __len__(self) -> int:
        return len(self.per_entity)


def _days_since(date: pd.Timestamp, origin: pd.Timestamp) -> float:
    return float((pd.Timestamp(date) - pd.Timestamp(origin)).days)


def _elapsed(dates: pd.Series, origin: pd.Timestamp) -> pd.Series:
    return (pd.to_datetime(dates) - pd.Timestamp(origin)).dt.days.astype("float64")


def fit(frame: pd.DataFrame, *, origin: pd.Timestamp | None = None) -> StoreTrends:
    """Fit one line per store from training rows. Closed days must already be dropped.

    `origin` anchors `t = 0`. It defaults to the earliest row given, and must be carried
    into serving unchanged — an intercept measured from one origin and applied from
    another is wrong by the slope times the gap between them, silently.
    """
    if frame.empty:
        raise StockoutError("cannot fit a trend on an empty frame")

    dates = pd.to_datetime(frame[s.DATE])
    anchor = pd.Timestamp(origin) if origin is not None else pd.Timestamp(dates.min())
    elapsed = _elapsed(dates, anchor)

    global_line = _line(elapsed.to_numpy(), frame[s.SALES].to_numpy())

    per_entity: dict[str, tuple[float, float]] = {}
    fell_back: set[str] = set()
    for entity, group in frame.groupby(frame[s.ENTITY].astype("string"), sort=False):
        key = str(entity)
        if len(group) < MIN_ROWS_PER_ENTITY:
            fell_back.add(key)
            continue
        span = _elapsed(group[s.DATE], anchor).to_numpy()
        # A store whose rows all fall on one date has no slope to fit; `polyfit` would
        # return a singular-matrix warning and a meaningless pair.
        if np.ptp(span) == 0:
            fell_back.add(key)
            continue
        per_entity[key] = _line(span, group[s.SALES].to_numpy())

    return StoreTrends(per_entity, global_line, anchor, frozenset(fell_back))


def _line(elapsed: np.ndarray, sales: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(elapsed, sales, 1)
    return float(intercept), float(slope)


def levels(frame: pd.DataFrame, trends: StoreTrends) -> pd.Series:
    """Each row's place on its own store's line."""
    entities = frame[s.ENTITY].astype("string")
    elapsed = _elapsed(frame[s.DATE], trends.origin)

    intercepts = entities.map(lambda e: trends.line_for(str(e))[0]).astype("float64")
    slopes = entities.map(lambda e: trends.line_for(str(e))[1]).astype("float64")
    return intercepts + slopes * elapsed


def remove(frame: pd.DataFrame, trends: StoreTrends) -> pd.Series:
    """The target with each store's line subtracted — what the tree is actually fitted on.

    What is left is *how unusual was this day for this store, given where its level had
    got to* — a question with no time direction in it, which is exactly the question a
    tree can answer and keep answering outside the window it was fitted on.
    """
    return frame[s.SALES].astype("float64") - levels(frame, trends)


def restore(residuals: np.ndarray, entities: pd.Series, dates: pd.Series,
            trends: StoreTrends) -> np.ndarray:
    """Put the line back under the tree's answer."""
    frame = pd.DataFrame({s.ENTITY: entities.astype("string").to_numpy(),
                          s.DATE: pd.to_datetime(dates).to_numpy()})
    return np.asarray(residuals, dtype="float64") + levels(frame, trends).to_numpy()


def horizon_warning(date: pd.Timestamp, observed_through: pd.Timestamp) -> str | None:
    """A sentence for a date far enough out that the line has stopped meaning much."""
    days = _days_since(date, observed_through)
    if days <= MAX_CONFIDENT_DAYS:
        return None
    return (
        f"{days:,.0f} days past the data. The trend is a straight line fitted on "
        f"2.5 years and extended; this far out it is arithmetic rather than a forecast."
    )
