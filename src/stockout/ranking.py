"""Rank every item a shop stocks for one day — the question a product model exists for.

`predict.py` answers one series at a time: *how many of item 25 will store 3 shift on
Tuesday*. That is the right unit for a model and the wrong unit for a person. Nobody
orders one item; they order a shelf, and what they need is the whole list sorted, with the
busy ones at the top and each one judged against its own usual.

So this scores every item in one pass rather than looping `predict`. Fifty single-row
builds produce the same numbers and pay the frame-construction cost fifty times over; the
arithmetic was never the expensive part. `predict.serving_matrix` builds one matrix, each
model answers once, and the per-series lines go back underneath in a single vectorised
addition.

**Every figure is in units, judged per series.** A shop's busiest item and its quietest are
not comparable on a shared scale, so `demand_class` here means the same thing it means
everywhere else in this package: busy *for this item in this shop*, against cut points
fitted on the training rows alone. A list sorted by predicted units puts the big sellers on
top; the class beside each one says whether that number is unusual for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import targets, trend
from .data import schemas as s
from .errors import StockoutError
from .persistence import Artifact
from .predict import date_status, serving_matrix
from .window import DateStatus


@dataclass(frozen=True)
class RankedItem:
    """One item's answer, with what it takes for that number to be unusual."""

    entity: str
    item: str
    sales: float
    demand_class: str
    thresholds: tuple[float, float]

    #: What this item normally does **on this date** — its own fitted line, read at the
    #: requested day. Falls back to its median across the window where the artefact has no
    #: lines, and to `None` where the caller supplied nothing.
    #:
    #: The line, not the median, and the difference is the whole point. `demand_class`
    #: above is cut on *residuals* — distance from this line — so a median baseline puts
    #: the two side by side measuring against different things. On a growing item in the
    #: seasonal peak that produced rows reading `Low` beside `+52% vs its usual`, both
    #: technically right and together plainly nonsense: the item was far above its
    #: all-time median and slightly below where its own line said it should be that day.
    typical: float | None = None

    @property
    def difference_percent(self) -> float:
        """How far above or below this item's own line for the day the prediction sits."""
        if not self.typical:
            return 0.0
        return (self.sales - self.typical) / self.typical * 100.0


@dataclass(frozen=True)
class Ranking:
    """Every item in one shop on one day, busiest first."""

    store: str
    date: pd.Timestamp
    items: list[RankedItem]
    status: DateStatus | None = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def total(self) -> float:
        """Predicted units across the whole shelf, which is what an order adds up to."""
        return float(sum(item.sales for item in self.items))

    def busiest(self, count: int) -> list[RankedItem]:
        return self.items[:count]


def items_in(artifact: Artifact, store: str) -> list[str]:
    """Every entity belonging to one store, in the artefact's own order.

    Read off the fitted entity list rather than the data: these are the series the model
    can actually answer about, which is not always every pair the source contains.
    """
    prefix = f"{store}{s.PRODUCT_ENTITY_SEPARATOR}"
    return [entity for entity in artifact.entities if entity.startswith(prefix)]


def stores_in(artifact: Artifact) -> list[str]:
    """The distinct store ids across the fitted entities, in numeric order where possible."""
    seen = {entity.split(s.PRODUCT_ENTITY_SEPARATOR, 1)[0] for entity in artifact.entities}
    return sorted(seen, key=_sortable)


def _sortable(value: str) -> tuple[int, float, str]:
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)


def rank(
    artifact: Artifact,
    *,
    store: str,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
    typical: dict[str, float] | None = None,
) -> Ranking:
    """Every item in one shop on one day, busiest first.

    `typical` is an optional per-entity median so each row can be compared with its own
    usual day. Passed in rather than derived here because the frame it comes from lives in
    the web service, and this module deliberately holds nothing but the artefact.
    """
    if not artifact.has_products:
        raise StockoutError(
            f"{artifact.dataset} has no product axis; it records one row per "
            f"{artifact.entity_label}-day, so there is no shelf to rank"
        )

    entities = items_in(artifact, str(store))
    if not entities:
        known = ", ".join(stores_in(artifact)[:_STORES_SHOWN])
        raise StockoutError(f"no items recorded for store {store!r}; known stores: {known}, ...")

    matrix = serving_matrix(
        artifact,
        entities=entities,
        date=date,
        promo=promo,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )

    stamp = pd.Timestamp(pd.to_datetime(date))
    sales = _restored(artifact, artifact.regressor.predict(matrix), entities, stamp)
    classes = [str(label) for label in artifact.classifier.predict(matrix)]

    items = [
        RankedItem(
            entity=entity,
            item=entity.split(s.PRODUCT_ENTITY_SEPARATOR, 1)[1],
            sales=float(predicted),
            demand_class=demand,
            thresholds=_thresholds_at(artifact, entity, stamp),
            typical=_usual_for(artifact, entity, stamp, typical),
        )
        for entity, predicted, demand in zip(entities, sales, classes, strict=True)
    ]
    items.sort(key=lambda item: item.sales, reverse=True)

    return Ranking(
        store=str(store), date=stamp, items=items, status=date_status(artifact, date)
    )


def _usual_for(
    artifact: Artifact,
    entity: str,
    stamp: pd.Timestamp,
    fallback: dict[str, float] | None,
) -> float | None:
    """What this item normally does on this date, on the same scale as its cut points.

    The item's own fitted line where the artefact has one. That is the baseline the demand
    bands are cut against — they are terciles of the *residual* — so measuring the printed
    percentage against anything else puts two columns beside each other that answer
    different questions and appear to contradict.

    The supplied medians are the fallback for an artefact with no lines. They are a poorer
    baseline even then: a median over the whole window sits below a growing item's level
    late in that window, so every late date reads as unusually busy.
    """
    if artifact.trends is not None:
        return float(artifact.trends.level_at(entity, stamp))
    return None if fallback is None else fallback.get(entity)


def _restored(
    artifact: Artifact, residuals: np.ndarray, entities: list[str], stamp: pd.Timestamp
) -> np.ndarray:
    """Put every series' own line back under the tree's answer, in one addition.

    The regressor answers in residuals on a de-trended artefact. Without this the list
    would rank correctly and print numbers near zero, which is not a unit count.
    """
    if artifact.trends is None:
        return np.asarray(residuals, dtype="float64")
    return trend.restore(
        np.asarray(residuals, dtype="float64"),
        pd.Series(entities, dtype="string"),
        pd.Series([stamp] * len(entities)),
        artifact.trends,
    )


def _thresholds_at(
    artifact: Artifact, entity: str, stamp: pd.Timestamp
) -> tuple[float, float]:
    """This series' cut points, shifted onto its line so they read in units.

    The same correction `predict._thresholds_at` makes, and for the same reason: stored
    cut points are residuals on a de-trended artefact, so printed raw they come out
    negative beside a positive unit count.
    """
    low, high = artifact.thresholds.for_entity(entity)
    if artifact.trends is None:
        return low, high
    level = artifact.trends.level_at(entity, stamp)
    return low + level, high + level


def classify(artifact: Artifact, sales: float, entity: str) -> str:
    """The class a predicted count falls into, by that series' own cut points."""
    return targets.classify_one(sales, str(entity), artifact.thresholds)


#: How many store ids an error lists before giving up and trailing off.
_STORES_SHOWN = 5
