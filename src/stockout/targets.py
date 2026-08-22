"""The demand class, and where its cut points come from.

Regression predicts `sales` directly; nothing to define. Classification predicts whether a
day is Low, Medium or High demand, and that phrase means nothing until somebody fixes the
cut points. Two decisions make it one.

**Per entity, not global.** Global terciles would put almost every day of a busy store in
High and almost every day of a quiet one in Low, so the classifier would learn store
identity, score around 95%, and know nothing about demand. Per-entity terciles subtract
the store's own level and leave the question worth asking: *is today busy for this shop?*

**Fitted on training rows only.** A tercile cut point is a statistic. One computed over
the whole frame has let the test rows vote on their own labels, and the resulting accuracy
is not a number anybody can defend. `fit` is given the training slice and nothing else.

The cut points travel in the artefact because a predicted number cannot be turned into
Low, Medium or High without them, and they are a property of the training window rather
than of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import schemas as s
from .errors import StockoutError

#: In the order a human reads them, stated explicitly. `LabelEncoder` sorts
#: alphabetically, which would map High -> 0 and Low -> 1 and caption every table
#: wrongly while looking entirely plausible.
DEMAND_CLASSES: tuple[str, ...] = ("Low", "Medium", "High")

LOW, MEDIUM, HIGH = DEMAND_CLASSES

#: The tercile boundaries.
_LOWER_QUANTILE = 1 / 3
_UPPER_QUANTILE = 2 / 3

#: An entity needs at least this many training rows for its own quantiles to mean
#: anything. Below it the global cut points are used and the fallback is recorded.
MIN_ROWS_PER_ENTITY = 30


@dataclass(frozen=True)
class DemandThresholds:
    """Per-entity cut points, with a global pair for entities that had too few rows."""

    per_entity: dict[str, tuple[float, float]]
    global_cuts: tuple[float, float]
    fell_back: frozenset[str] = field(default_factory=frozenset)

    def for_entity(self, entity: str) -> tuple[float, float]:
        return self.per_entity.get(str(entity), self.global_cuts)

    def describe(self, entity: str) -> str:
        low, high = self.for_entity(entity)
        return f"Low <{low:,.0f} | Medium {low:,.0f}-{high:,.0f} | High >{high:,.0f}"

    def __len__(self) -> int:
        return len(self.per_entity)


def fit(frame: pd.DataFrame) -> DemandThresholds:
    """Compute cut points from training rows. Closed days must already be dropped."""
    if frame.empty:
        raise StockoutError("cannot fit demand thresholds on an empty frame")

    sales = frame[s.SALES]
    global_cuts = (
        float(sales.quantile(_LOWER_QUANTILE)),
        float(sales.quantile(_UPPER_QUANTILE)),
    )

    per_entity: dict[str, tuple[float, float]] = {}
    fell_back: set[str] = set()
    for entity, group in frame.groupby(frame[s.ENTITY].astype("string"), sort=False):
        key = str(entity)
        if len(group) < MIN_ROWS_PER_ENTITY:
            fell_back.add(key)
            continue
        low = float(group[s.SALES].quantile(_LOWER_QUANTILE))
        high = float(group[s.SALES].quantile(_UPPER_QUANTILE))
        # A store whose quantiles collapse sells the same amount every day; its own
        # cut points would put every row in one class, so the global pair is better.
        if low == high:
            fell_back.add(key)
            continue
        per_entity[key] = (low, high)

    return DemandThresholds(per_entity, global_cuts, frozenset(fell_back))


def apply(frame: pd.DataFrame, thresholds: DemandThresholds) -> pd.Series:
    """Label every row Low, Medium or High against its own entity's cut points."""
    entities = frame[s.ENTITY].astype("string")
    lows = entities.map(lambda e: thresholds.for_entity(str(e))[0])
    highs = entities.map(lambda e: thresholds.for_entity(str(e))[1])

    labels = pd.Series(MEDIUM, index=frame.index, dtype="object")
    labels[frame[s.SALES] <= lows] = LOW
    labels[frame[s.SALES] > highs] = HIGH
    return labels.astype("string")


def classify_one(sales: float, entity: str, thresholds: DemandThresholds) -> str:
    """The same rule for a single predicted number, which is what serving needs."""
    low, high = thresholds.for_entity(entity)
    if sales <= low:
        return LOW
    if sales > high:
        return HIGH
    return MEDIUM


def class_balance(labels: pd.Series) -> dict[str, float]:
    """Share of each class. Per-entity terciles make this near 1/3 by construction, and
    that is the floor any classifier has to clear to have learnt anything."""
    counts = labels.value_counts(normalize=True)
    return {name: float(counts.get(name, 0.0)) for name in DEMAND_CLASSES}
