"""What the deployed model can be asked about: the entities, and the shelf behind a shop.

A store-day deployment offers one list — pick a shop. A product deployment offers two —
pick a shop, then pick an item — because a single dropdown of every store-item pair is
five hundred entries where two dropdowns are ten and fifty.

Kept apart from `service.py` so that *what can be asked* and *what the answer is* do not
share a file. This module only lists and describes; nothing here predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockout.data import schemas as s
from stockout.datasets import Prepared
from stockout.features import ROW_FEATURES
from stockout.persistence import Artifact
from stockout.ranking import items_in, stores_in

from .presentation import StoreProfile


@dataclass(frozen=True)
class Catalogue:
    """The entities one deployed artefact can answer about, described."""

    artifact: Artifact
    profiles: dict[str, StoreProfile]

    @property
    def has_products(self) -> bool:
        return self.artifact.has_products

    def uses_promotions(self) -> bool:
        """Whether the deployed model reads promotions and holidays at all.

        Asked of the fitted matrix rather than of the source, because the matrix is what
        the deployed artefact actually carries. A product source records none of these, so
        `feature_columns` dropped them and a form offering the controls would be inviting
        input that cannot change an answer.
        """
        return any(column in self.artifact.feature_columns for column in ROW_FEATURES)

    def profile_for(self, entity: str) -> StoreProfile | None:
        return self.profiles.get(str(entity))

    def entity_options(self) -> list[StoreProfile]:
        """Every entity, quietest first, so the list reads as a range.

        The store-day form's only dropdown. On a product deployment this would be one
        entry per store-item pair, which is why `item_options` exists instead.
        """
        return sorted(self.profiles.values(), key=lambda profile: profile.typical_sales)

    def stores(self) -> list[str]:
        """The distinct shop ids, in numeric order where the ids allow it."""
        if not self.has_products:
            return list(self.artifact.entities)
        return stores_in(self.artifact)

    def item_options(self, store: str) -> list[StoreProfile]:
        """Every item one shop stocks, busiest first.

        Busiest first rather than by id: the top of this list is what the shop actually
        sells, and an id order would open on whichever item happens to be numbered 1.
        """
        profiles = [
            self.profiles[entity]
            for entity in items_in(self.artifact, str(store))
            if entity in self.profiles
        ]
        return sorted(profiles, key=lambda profile: profile.typical_sales, reverse=True)

    def categories(self) -> list[str]:
        """The kinds of thing this deployment stocks, in the order they were catalogued.

        Read off the fitted profiles rather than the vocabulary: the vocabulary is every
        category the encoder can represent, and this is the set the deployed model has
        actually seen rows of.
        """
        seen = {p.category for p in self.profiles.values() if p.category}
        return sorted(seen)

    def entity_for(self, store: str, item: str | None) -> str:
        """The series id from the two halves a product form submits."""
        if not self.has_products:
            return str(store)
        return f"{store}{s.PRODUCT_ENTITY_SEPARATOR}{item}"

    def typical(self) -> dict[str, float]:
        """Each entity's median day, for comparing a prediction with its own usual.

        Read off the profiles rather than recomputed: they already hold it, and the
        ranked page wants the whole mapping at once.
        """
        return {entity: profile.typical_sales for entity, profile in self.profiles.items()}


def actual_for(prepared: Prepared, entity: str, date: str | pd.Timestamp) -> float | None:
    """What the entity really recorded that day, where the dataset has a row for it.

    This is the honest reply to the obvious objection: inside the fitted window a
    prediction is redundant, because the answer is already in the file. Rather than hide
    that, the panel shows both and lets the error stand next to the guess — which turns
    the redundancy from an embarrassment into the only place on the site where a visitor
    can watch the model be checked against something it cannot argue with.

    `None` for a day with no row: dates past the data, and closed days, which
    `features.drop_closed` removes before anything is fitted.
    """
    frame = prepared.frame
    stamp = pd.to_datetime(date)
    rows = frame[
        (frame[s.ENTITY].astype("string") == str(entity))
        & (pd.to_datetime(frame[s.DATE]) == stamp)
    ]
    if rows.empty:
        return None
    return float(rows[s.SALES].iloc[0])
