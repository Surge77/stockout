"""Answer one store-day: how much it will sell, and whether that is Low, Medium or High.

Both answers come from the artefact, and they come from two different models that can
disagree. Showing one while implying both would be the most misleading thing here, so
`Prediction` always carries the pair and the cut points that separate the classes.

The single serving row is built through exactly the same `features.build` the training
frame went through, with the encoders the artefact carries. That is the whole reason the
encoders travel: a second `LabelEncoder` fitted here would renumber the store types and
answer confidently with nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import features, targets
from .data import schemas as s
from .errors import StockoutError
from .persistence import Artifact
from .registry import model_names

#: A serving row has no sales yet. The column has to exist for the feature builder, and
#: nothing reads it — every feature is built from the calendar and the store metadata.
_SALES_PLACEHOLDER = 0.0


@dataclass
class Prediction:
    """What the panel and the CLI both display."""

    entity: str
    date: pd.Timestamp
    sales: float
    demand_class: str
    thresholds: tuple[float, float]
    per_model: dict[str, float] = field(default_factory=dict)

    def threshold_text(self) -> str:
        low, high = self.thresholds
        return f"Low <{low:,.0f} | Medium {low:,.0f}-{high:,.0f} | High >{high:,.0f}"

    def lines(self) -> list[str]:
        return [
            f"{self.entity} on {self.date:%Y-%m-%d}",
            f"  Predicted sales   {self.sales:,.0f}",
            f"  Demand class      {self.demand_class}",
            f"  cut points        {self.threshold_text()}",
        ]


def serving_row(
    artifact: Artifact,
    *,
    entity: str,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> pd.DataFrame:
    """Build the one-row model matrix for a store-day, exactly as training built it."""
    entity = str(entity)
    if artifact.entities and entity not in artifact.entities:
        raise StockoutError(
            f"unknown {artifact.entity_label} {entity!r}; the model was fitted on "
            f"{len(artifact.entities):,} of them"
        )

    stamp = pd.to_datetime(date)
    if pd.isna(stamp):
        raise StockoutError(f"{date!r} is not a date")

    row = pd.DataFrame(
        [
            {
                s.ENTITY: entity,
                s.DATE: stamp,
                s.SALES: _SALES_PLACEHOLDER,
                s.OPEN: 1,
                s.PROMO: int(promo),
                s.STATE_HOLIDAY: state_holiday,
                s.SCHOOL_HOLIDAY: int(school_holiday),
            }
        ]
    )
    built, _ = features.build(row, artifact.stores, encoders=artifact.encoders)

    missing = [c for c in artifact.feature_columns if c not in built.columns]
    if missing:
        raise StockoutError(f"serving row is missing {', '.join(missing)}")
    return built[artifact.feature_columns]


def predict(
    artifact: Artifact,
    *,
    entity: str,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
    all_models: bool = False,
) -> Prediction:
    """Both answers for one store-day, optionally with every regressor's own number."""
    matrix = serving_row(
        artifact,
        entity=entity,
        date=date,
        promo=promo,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )

    sales = float(artifact.regressor.predict(matrix)[0])
    demand = str(artifact.classifier.predict(matrix)[0])
    entity = str(entity)

    return Prediction(
        entity=entity,
        date=pd.to_datetime(date),
        sales=sales,
        demand_class=demand,
        thresholds=artifact.thresholds.for_entity(entity),
        per_model=_every_regressor(artifact, matrix) if all_models else {},
    )


def _every_regressor(artifact: Artifact, matrix: pd.DataFrame) -> dict[str, float]:
    """What each registered regressor predicts for this row.

    Only the deployed model is refitted-and-ready; the others are not in the artefact, so
    this reports the deployed one and leaves the rest to the caller that has a training
    frame. The panel uses `compare_row` below instead.
    """
    return {artifact.regressor_name: float(artifact.regressor.predict(matrix)[0])}


def compare_row(
    artifact: Artifact,
    fitted: dict[str, object],
    *,
    entity: str,
    date: str | pd.Timestamp,
    promo: bool = False,
) -> dict[str, float]:
    """Every fitted regressor's prediction for one row, for the panel's comparison panel.

    `fitted` comes from the web service, which keeps the whole registry fitted in memory
    so the expandable section can show that the models disagree.
    """
    matrix = serving_row(artifact, entity=entity, date=date, promo=promo)
    answers: dict[str, float] = {}
    for name in model_names("regression"):
        model = fitted.get(name)
        if model is not None:
            answers[name] = float(model.predict(matrix)[0])  # type: ignore[attr-defined]
    return answers


def classify(artifact: Artifact, sales: float, entity: str) -> str:
    """Turn a predicted number into a class with that entity's own cut points."""
    return targets.classify_one(sales, str(entity), artifact.thresholds)
