"""What one entity looks like, so a dropdown entry and a comparison mean something.

A dropdown of bare ids tells a reader nothing. A profile turns each one into a sentence:
what kind of shop it is, or which item in which shop, and what an ordinary day there looks
like — the context every number further down the page is relative to.

Two shapes, because the sources have two. A store-day entity is described by the metadata
Rossmann publishes about it. A store-item entity has no metadata at all, so it is described
by the only things that exist: which shop, which item, and its own usual day.

Nothing here computes anything new. It renames and phrases what the models already
produced. `verdict.py` holds the phrasing of a single prediction; this holds the entities.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockout.data import schemas as s

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

_KM = 1000.0
_PERCENT = 100.0


@dataclass(frozen=True)
class StoreProfile:
    """What one entity looks like, so a dropdown entry means something."""

    entity: str
    store_type: str
    assortment: str
    typical_sales: float
    quiet_cut: float
    busy_cut: float
    competition_km: float | None
    has_promo2: bool
    days: int

    #: The two halves of a store-item entity, empty on a store-day one. Split once here
    #: rather than in each template that wants to say "item 7 in store 3".
    store: str = ""
    item: str = ""

    #: What the item is, where the source publishes a catalogue. Empty on
    #: `store_item_demand`, which gives an integer per item and nothing else — so a page
    #: that wants to name an item has to cope with there being no name, and does.
    product_name: str = ""
    category: str = ""

    #: What `typical_sales` counts, so the wording can say "units" where that is the truth
    #: and stay silent where the figure is a till total in an unstated currency.
    unit: str = "currency"

    @property
    def is_product(self) -> bool:
        return bool(self.item)

    @property
    def title(self) -> str:
        """What to call this entity on screen.

        The catalogue's name where there is one, the bare id where there is not. Both are
        real states: this repository generates `synthetic_products` and can therefore say
        what item 7 is, while Kaggle's file publishes the integer alone and inventing a
        name for it would be the fabrication this project refuses elsewhere.
        """
        if self.product_name:
            return self.product_name
        if self.is_product:
            return f"Item {self.item}"
        return f"Shop {self.entity}"

    @property
    def amount_text(self) -> str:
        """`typical_sales` with its unit, where there is one worth naming."""
        if self.unit == "units":
            return f"{self.typical_sales:,.0f} units"
        return f"{self.typical_sales:,.0f}"

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
        """The text in the dropdown. A bare id tells a reader nothing.

        A product source publishes nothing describing an item — no name, no category, no
        price — so there is no honest way to say what item 7 *is*. What can be said is
        what it does: how much of it this shop shifts on an ordinary day, which is also
        the figure every prediction below is relative to.
        """
        if self.is_product:
            kind = f" ({self.category})" if self.category else ""
            return f"{self.title}{kind} — about {self.amount_text} on a normal day here"
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


def build_profiles(
    frame: pd.DataFrame,
    thresholds,
    stores: pd.DataFrame | None,
    *,
    has_products: bool = False,
    unit: str = "currency",
) -> dict[str, StoreProfile]:
    """One profile per entity, from the prepared frame and whatever metadata exists.

    `has_products` comes from the source rather than from the shape of the ids, for the
    same reason `features.add_product_features` takes it: a store-day source's behaviour
    must not depend on how its ids happen to be spelled.
    """
    metadata = _metadata_lookup(stores)
    profiles: dict[str, StoreProfile] = {}

    grouped = frame.groupby(frame[s.ENTITY].astype("string"), sort=False)
    for entity, group in grouped:
        key = str(entity)
        quiet, busy = thresholds.for_entity(key)
        facts = metadata.get(key, {})
        store, item = _split_entity(key) if has_products else ("", "")
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
            store=store,
            item=item,
            product_name=str(facts.get(s.PRODUCT_NAME, "") or ""),
            category=str(facts.get(s.CATEGORY, "") or ""),
            unit=unit,
        )
    return profiles


def _split_entity(entity: str) -> tuple[str, str]:
    store, _, item = entity.partition(s.PRODUCT_ENTITY_SEPARATOR)
    return store, item


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
