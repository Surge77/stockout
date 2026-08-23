"""Answer about a shop that does not exist yet, described by its attributes alone.

The rest of the package answers *how much will store 12 take on Monday*. That question
comes with an id, and the deployed regressor uses it: `entity_code` is one of its inputs.

This module answers a different question — *how much would a shop like this take* — and
it cannot use the deployed regressor to do it. A store the fit has never seen has no
`entity_code` to supply, and inventing one would put the prediction somewhere in the
middle of stores it has nothing to do with. `Artifact.cold_regressor` exists for exactly
this: the same estimator, fitted on every feature except identity.

Whether the answer is worth anything is a property of the data, not of this code. It only
works because a store's attributes genuinely move its sales — assortment, store type and
distance to the nearest competitor all feed the level in `data/synth.py`. Before that was
true, holding a store out of training and predicting it scored r² -0.70, worse than
guessing that store's own average, and this module would have returned confident nonsense.
`Artifact.cold_scores["unseen_store_r2"]` carries the leave-one-store-out score so the
page can show what the estimate is actually worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from . import features, targets, window
from .data import schemas as s
from .errors import StockoutError
from .persistence import Artifact
from .predict import date_status
from .window import DateStatus

#: A serving row needs the column to exist; nothing reads it.
_SALES_PLACEHOLDER = 0.0

#: What a described store leaves unsaid. A hypothetical shop has no competitor opening
#: date and no Promo2 start week, which is the same shape as a real store missing them —
#: the feature builder already turns both into a zero age.
_UNKNOWN_HISTORY: dict[str, object] = {
    s.COMPETITION_OPEN_MONTH: float("nan"),
    s.COMPETITION_OPEN_YEAR: float("nan"),
    s.PROMO2_SINCE_WEEK: float("nan"),
    s.PROMO2_SINCE_YEAR: float("nan"),
    s.PROMO_INTERVAL: "",
}

#: The id given to the described store. Never reaches the model — `cold_feature_columns`
#: excludes `entity_code` — but `features.build` needs the column present.
_PLACEHOLDER_ENTITY = "hypothetical"


@dataclass(frozen=True)
class StoreDescription:
    """What somebody types in to describe a shop."""

    store_type: str
    assortment: str
    competition_distance: float | None
    promo2: bool

    def as_row(self) -> dict[str, object]:
        return {
            s.STORE_TYPE: self.store_type,
            s.ASSORTMENT: self.assortment,
            s.COMPETITION_DISTANCE: (
                float("nan")
                if self.competition_distance is None
                else float(self.competition_distance)
            ),
            s.PROMO2: int(self.promo2),
            **_UNKNOWN_HISTORY,
        }


@dataclass
class HypotheticalPrediction:
    """Predicted sales for a described store, and how much to trust it."""

    sales: float
    demand_class: str
    thresholds: tuple[float, float]
    unseen_store_r2: float
    description: StoreDescription
    date: pd.Timestamp
    per_model: dict[str, float] = field(default_factory=dict)

    #: As on `Prediction`. A described store has no actual in either region — it does not
    #: exist — so this says only whether the *calendar* position is one the fit covered.
    status: DateStatus | None = None


def predict_described(
    artifact: Artifact,
    description: StoreDescription,
    *,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> HypotheticalPrediction:
    """Sales for a store that does not exist, on one day.

    The demand class is derived from the *global* cut points rather than a store's own,
    because a described store has no history to take terciles of. That is a real
    limitation and the page says so rather than implying the class is per-store.
    """
    if artifact.cold_regressor is None:
        raise StockoutError(
            "this artefact has no cold-start model; retrain it with the current build"
        )

    matrix = _serving_row(
        artifact, description, date=date, promo=promo,
        state_holiday=state_holiday, school_holiday=school_holiday,
    )
    residual = float(artifact.cold_regressor.predict(matrix)[0])
    # A described shop has no line of its own, so the chain-wide one carries the level.
    # `level_at` falls back to `global_line` for an entity it has never seen, and the
    # placeholder id is exactly that. The reading is *where the chain is on that date,
    # plus how far a shop like this sits from it* — which also means this page moves with
    # the date past the end of the data, where before it could not.
    sales = residual
    if artifact.trends is not None:
        sales += artifact.trends.level_at(_PLACEHOLDER_ENTITY, pd.to_datetime(date))
    low, high = artifact.thresholds.global_cuts

    return HypotheticalPrediction(
        sales=sales,
        demand_class=_class_for(sales, low, high),
        thresholds=(low, high),
        unseen_store_r2=float(artifact.cold_scores.get("unseen_store_r2", 0.0)),
        description=description,
        date=pd.to_datetime(date),
        status=date_status(artifact, date),
    )


def contributions(
    artifact: Artifact,
    description: StoreDescription,
    *,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> list[tuple[str, float]]:
    """What each described attribute is worth, in currency, for this one prediction.

    One-at-a-time counterfactuals: hold everything else fixed, move one input to a plain
    baseline, and report how far the answer moves. **Not** an additive decomposition —
    the parts will not sum to the total, because a tree ensemble is not additive and
    these inputs interact. It answers "what is this attribute doing here", which is the
    question a reader actually has, and the page says which question it answered.
    """
    actual = predict_described(
        artifact, description, date=date, promo=promo,
        state_holiday=state_holiday, school_holiday=school_holiday,
    ).sales

    def swap(**changes: object) -> StoreDescription:
        return replace(description, **changes)  # type: ignore[arg-type]

    # Each baseline is the plainest version of that one attribute: the basic range, the
    # middle store type, no recorded competitor, no continuing promotion.
    baselines = {
        "Assortment": swap(assortment="a"),
        "Store type": swap(store_type="c"),
        "Competitor distance": swap(competition_distance=None),
        "Promo2": swap(promo2=False),
    }

    out: list[tuple[str, float]] = []
    for label, alternative in baselines.items():
        counterfactual = predict_described(
            artifact, alternative, date=date, promo=promo,
            state_holiday=state_holiday, school_holiday=school_holiday,
        ).sales
        out.append((label, actual - counterfactual))

    without_promo = predict_described(
        artifact, description, date=date, promo=False,
        state_holiday=state_holiday, school_holiday=school_holiday,
    ).sales
    out.append(("Promotion", actual - without_promo))

    return sorted(out, key=lambda pair: abs(pair[1]), reverse=True)


def _serving_row(
    artifact: Artifact,
    description: StoreDescription,
    *,
    date: str | pd.Timestamp,
    promo: bool,
    state_holiday: str,
    school_holiday: bool,
) -> pd.DataFrame:
    # Same window policy as the identified-store path. No age cap is needed here: the
    # cold model excludes the three age features by construction, and a described store
    # supplies `_UNKNOWN_HISTORY` for them regardless, so nothing monotone reaches it.
    status = date_status(artifact, date)
    stamp = status.stamp if status else window.parse(date)

    row = pd.DataFrame(
        [
            {
                s.ENTITY: _PLACEHOLDER_ENTITY,
                s.DATE: stamp,
                s.SALES: _SALES_PLACEHOLDER,
                s.OPEN: 1,
                s.PROMO: int(promo),
                s.STATE_HOLIDAY: state_holiday,
                s.SCHOOL_HOLIDAY: int(school_holiday),
            }
        ]
    )
    described = pd.DataFrame([{s.ENTITY: _PLACEHOLDER_ENTITY, **description.as_row()}])
    built, _ = features.build(row, described, encoders=artifact.encoders)

    missing = [c for c in artifact.cold_feature_columns if c not in built.columns]
    if missing:
        raise StockoutError(f"described row is missing {', '.join(missing)}")
    return built[artifact.cold_feature_columns]


def _class_for(sales: float, low: float, high: float) -> str:
    if sales < low:
        return targets.LOW
    if sales > high:
        return targets.HIGH
    return targets.MEDIUM
