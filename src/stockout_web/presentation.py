"""Turning model output into words a shopkeeper can act on.

The library speaks in r², terciles and class labels. A person looking at the page wants to
know whether to order more stock. This module is the whole translation, kept out of
`service.py` so that the modelling and the wording can be changed independently.

Nothing here computes anything new. It renames, compares and phrases what the models
already produced — and where it makes a claim about how good a number is, that claim comes
from the held-out score rather than from an adjective.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockout.data import schemas as s
from stockout.targets import HIGH, LOW, MEDIUM

#: Rossmann's own documented meaning for the assortment codes. `store_type` a/b/c/d is
#: anonymised in the source and has no published meaning, so it is shown as a bare letter
#: rather than given an invented label.
#: How much variety a shop stocks. These three meanings are documented by the competition,
#: so the words are the source's own rather than a guess — unlike the store grouping below.
ASSORTMENT_LABELS: dict[str, str] = {
    "a": "stocks a basic range",
    "b": "stocks an extra range",
    "c": "stocks an extended range",
}

#: The store grouping, spelled out rather than left as a letter.
#:
#: Rossmann sorted its shops into four kinds and **never published what the kinds mean**.
#: The letters are all the source gives. Inventing names for them — "Supermarket",
#: "Express" — would read better and be fiction, so the letter stays and the page explains
#: what it is instead: a grouping the shops' owner considered meaningful, which the model
#: uses because shops in the same group turn out to trade alike.
STORE_GROUP_LABELS: dict[str, str] = {
    "a": "shop group A",
    "b": "shop group B",
    "c": "shop group C",
    "d": "shop group D",
}

#: What each demand class means for somebody deciding how much to order.
DEMAND_WORDS: dict[str, dict[str, str]] = {
    LOW: {
        "title": "Quiet day",
        "advice": "Below this store's usual. Ordering the normal amount risks leftover stock.",
        "tone": "quiet",
    },
    MEDIUM: {
        "title": "Typical day",
        "advice": "An ordinary day for this store. The usual order should be about right.",
        "tone": "typical",
    },
    HIGH: {
        "title": "Busy day",
        "advice": "Above this store's usual. Order extra so you do not run out.",
        "tone": "busy",
    },
}

_KM = 1000.0
_PERCENT = 100.0

#: Headroom added to each end of the scale bar, as a fraction of its span. Enough that the
#: outer bands are always visible; small enough that the marker still sits where the
#: numbers say it should.
_AXIS_PADDING = 0.15


@dataclass(frozen=True)
class StoreProfile:
    """What one store looks like, so a dropdown entry means something."""

    entity: str
    store_type: str
    assortment: str
    typical_sales: float
    quiet_cut: float
    busy_cut: float
    competition_km: float | None
    has_promo2: bool
    days: int

    @property
    def assortment_label(self) -> str:
        return ASSORTMENT_LABELS.get(self.assortment.lower(), "range not recorded")

    @property
    def type_label(self) -> str:
        if not self.store_type:
            return "group not recorded"
        return STORE_GROUP_LABELS.get(self.store_type.lower(), "group not recorded")

    @property
    def option_label(self) -> str:
        """The text in the dropdown. A bare id tells a reader nothing."""
        return (
            f"Shop {self.entity} — {self.type_label}, {self.assortment_label}, "
            f"takes about {self.typical_sales:,.0f} on a normal day"
        )

    @property
    def competition_text(self) -> str:
        if self.competition_km is None:
            return "No competitor recorded nearby"
        if self.competition_km < 1:
            return f"Nearest competitor {self.competition_km * _KM:,.0f} m away"
        return f"Nearest competitor {self.competition_km:,.1f} km away"


@dataclass(frozen=True)
class Verdict:
    """One prediction, phrased."""

    sales: float
    demand_class: str
    profile: StoreProfile
    spread_low: float
    spread_high: float

    @property
    def title(self) -> str:
        return DEMAND_WORDS[self.demand_class]["title"]

    @property
    def advice(self) -> str:
        return DEMAND_WORDS[self.demand_class]["advice"]

    @property
    def tone(self) -> str:
        return DEMAND_WORDS[self.demand_class]["tone"]

    @property
    def implied_class(self) -> str:
        """The class the predicted *number* falls into, by this store's own cut points.

        Not the same thing as `demand_class`, which is what the classifier said. Two
        models, two answers, and nothing forces them to agree.
        """
        if self.sales < self.profile.quiet_cut:
            return LOW
        if self.sales > self.profile.busy_cut:
            return HIGH
        return MEDIUM

    @property
    def models_disagree(self) -> bool:
        return self.implied_class != self.demand_class

    @property
    def disagreement_text(self) -> str:
        """Said plainly, because the panel would otherwise contradict itself.

        `advice` is written from the classifier's label and `comparison_text` from the
        regressor's number. When the two models split, those two lines say opposite
        things one after the other — "below this store's usual" directly above "about the
        same as this store's usual" — which reads as a broken page rather than as the
        honest disagreement it is. Naming it turns a bug report into the point.
        """
        if not self.models_disagree:
            return ""
        return (
            f"The two models disagree here. {self.sales:,.0f} falls in "
            f"{self.implied_class} by this store's cut points, while the demand "
            f"classifier calls it {self.demand_class}. Neither corrects the other — "
            f"treat the day as borderline."
        )

    @property
    def difference_percent(self) -> float:
        if not self.profile.typical_sales:
            return 0.0
        return (self.sales - self.profile.typical_sales) / self.profile.typical_sales * _PERCENT

    @property
    def comparison_text(self) -> str:
        """How this compares with the store's own usual day, in one sentence."""
        difference = self.difference_percent
        usual = f"{self.profile.typical_sales:,.0f}"
        if abs(difference) < 5:
            return f"About the same as this store's usual {usual} a day."
        direction = "more than" if difference > 0 else "less than"
        return f"{abs(difference):,.0f}% {direction} this store's usual {usual} a day."

    @property
    def arrow(self) -> str:
        difference = self.difference_percent
        if abs(difference) < 5:
            return "="
        return "up" if difference > 0 else "down"

    def _axis(self) -> tuple[float, float]:
        """The range the bar spans, with room beyond both cut points.

        Taking the axis straight from `min`/`max` of the values collapses an outer band
        to zero width whenever nothing exceeds its cut point — which is the normal case.
        A prediction below `quiet_cut` puts `busy_cut` at exactly 100%, the busy band
        renders zero pixels wide, and the bar silently claims this store is never busy.
        The padding keeps all three bands on screen so the marker is read against a
        complete scale rather than a truncated one.
        """
        values = (
            self.sales,
            self.spread_low,
            self.spread_high,
            self.profile.quiet_cut,
            self.profile.busy_cut,
        )
        low, high = min(values), max(values)
        span = high - low
        padding = span * _AXIS_PADDING if span > 0 else max(abs(high) * _AXIS_PADDING, 1.0)
        return low - padding, high + padding

    def _position(self, value: float) -> float:
        """`value` as a 0-100 percentage of the axis."""
        low, high = self._axis()
        span = high - low
        if span <= 0:
            return 50.0
        return max(0.0, min(_PERCENT, (value - low) / span * _PERCENT))

    def scale_position(self) -> float:
        """Where the prediction sits on this store's own range, as a 0-100 percentage.

        The bar shows quiet / typical / busy as three bands and marks the prediction on
        them, which answers "is that a lot?" without the reader having to hold three
        numbers in their head.
        """
        return self._position(self.sales)

    def band_edges(self) -> tuple[float, float]:
        """The quiet/typical and typical/busy boundaries, as percentages of the bar."""
        return self._position(self.profile.quiet_cut), self._position(self.profile.busy_cut)

    @property
    def spread_text(self) -> str:
        width = self.spread_high - self.spread_low
        return (
            f"The models range from {self.spread_low:,.0f} to {self.spread_high:,.0f} "
            f"— a spread of {width:,.0f}."
        )


def build_profiles(
    frame: pd.DataFrame, thresholds, stores: pd.DataFrame | None
) -> dict[str, StoreProfile]:
    """One profile per store, from the prepared frame and the store metadata."""
    metadata = _metadata_lookup(stores)
    profiles: dict[str, StoreProfile] = {}

    grouped = frame.groupby(frame[s.ENTITY].astype("string"), sort=False)
    for entity, group in grouped:
        key = str(entity)
        quiet, busy = thresholds.for_entity(key)
        facts = metadata.get(key, {})
        profiles[key] = StoreProfile(
            entity=key,
            store_type=str(facts.get(s.STORE_TYPE, "") or ""),
            assortment=str(facts.get(s.ASSORTMENT, "") or ""),
            typical_sales=float(group[s.SALES].median()),
            quiet_cut=quiet,
            busy_cut=busy,
            competition_km=_km(facts.get(s.COMPETITION_DISTANCE)),
            has_promo2=bool(facts.get(s.PROMO2, 0)),
            days=len(group),
        )
    return profiles


def _metadata_lookup(stores: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if stores is None or s.ENTITY not in stores.columns:
        return {}
    indexed = stores.copy()
    indexed[s.ENTITY] = indexed[s.ENTITY].astype("string")
    return {str(row[s.ENTITY]): dict(row) for _, row in indexed.iterrows()}


def _km(distance: object) -> float | None:
    """Rossmann records competition distance in metres."""
    try:
        value = float(distance)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return value / _KM


def r2_in_words(score: float) -> str:
    """`r2` for somebody who has not met it before."""
    return (
        f"Explains about {score * _PERCENT:,.0f}% of why sales differ from day to day. "
        f"Higher is better; 100% would be perfect."
    )


def accuracy_in_words(score: float, floor: float) -> str:
    """Accuracy, against the floor it has to clear to mean anything."""
    correct = score * _PERCENT
    guessing = floor * _PERCENT
    return (
        f"Puts the day in the right band about {correct:,.0f} times in 100. "
        f"Always guessing the commonest band would score {guessing:,.0f}."
    )
