"""The described-shop question, and the model that is allowed to answer it.

*How much would a shop like this take* is not the question the deployed regressor answers.
That model reads `entity_code`, and a shop that does not exist has no id to supply — see
`stockout.hypothetical` for why inventing one puts the answer somewhere in the middle of
shops it has nothing to do with.

So this goes through `artifact.cold_regressor`, fitted on every feature except identity.
Where the source publishes nothing describing an entity that model is absent entirely, and
`ModelService.supports_description` is what the page reads to hide the form rather than
offer one that cannot be answered.
"""

from __future__ import annotations

from stockout.hypothetical import (
    HypotheticalPrediction,
    StoreDescription,
    contributions,
    predict_described,
)
from stockout.persistence import Artifact


def describe(
    artifact: Artifact,
    *,
    store_type: str,
    assortment: str,
    competition_distance: float | None,
    promo2: bool,
    date: str,
    promo: bool,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> tuple[HypotheticalPrediction, list[tuple[str, float]]]:
    """The answer for a described shop, and what each attribute was worth to it."""
    description = StoreDescription(
        store_type=store_type,
        assortment=assortment,
        competition_distance=competition_distance,
        promo2=promo2,
    )
    shared = {
        "date": date,
        "promo": promo,
        "state_holiday": state_holiday,
        "school_holiday": school_holiday,
    }
    answer = predict_described(artifact, description, **shared)
    why = contributions(artifact, description, **shared)
    return answer, why
