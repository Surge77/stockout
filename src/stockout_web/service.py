"""The one object the app keeps: the deployed models, and everything they need.

Built once at startup and held on `app.state`. Fitting takes seconds and the artefact is
tens of megabytes, so doing it per request would make every page load pay for it.

Retraining swaps the whole object rather than mutating it in place. A request that arrives
mid-retrain therefore sees either the old models or the new ones, never a half-swapped pair
where the regressor has been replaced and the thresholds have not.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stockout import persistence
from stockout import train as training
from stockout.datasets import Prepared, prepare
from stockout.errors import StockoutError
from stockout.evaluate import compare, split
from stockout.persistence import Artifact
from stockout.predict import Prediction, predict, serving_row
from stockout.registry import build, model_names

DEFAULT_DATASET = "synthetic"

#: Fitting all seven regressors so the user page can show that they disagree costs about
#: twenty seconds. Done once at startup and on retrain, never per request.
COMPARISON_MODELS: tuple[str, ...] = model_names("regression")


@dataclass
class ModelService:
    """The deployed artefact, the frame it came from, and the comparison fits."""

    artifact: Artifact
    prepared: Prepared
    others: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset: str = DEFAULT_DATASET) -> ModelService:
        """Read the saved artefact, or train one if this is a first run."""
        prepared = prepare(dataset)
        try:
            artifact = persistence.load()
        except StockoutError:
            artifact, _ = training.train(prepared)
        return cls(artifact=artifact, prepared=prepared, others=_fit_others(prepared))

    # -- reads -------------------------------------------------------------

    def entities(self) -> list[str]:
        return self.artifact.entities

    def entity_label(self) -> str:
        return self.artifact.entity_label

    def date_bounds(self) -> tuple[str, str]:
        start, end = self.prepared.date_range()
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def forecast(
        self,
        *,
        entity: str,
        date: str,
        promo: bool,
        school_holiday: bool = False,
    ) -> Prediction:
        """Both answers for one store-day, plus what every other regressor said."""
        answer = predict(
            self.artifact,
            entity=entity,
            date=date,
            promo=promo,
            school_holiday=school_holiday,
        )
        answer.per_model = self._every_model(
            entity=entity, date=date, promo=promo, school_holiday=school_holiday
        )
        return answer

    def _every_model(
        self, *, entity: str, date: str, promo: bool, school_holiday: bool
    ) -> dict[str, float]:
        matrix = serving_row(
            self.artifact,
            entity=entity,
            date=date,
            promo=promo,
            school_holiday=school_holiday,
        )
        return {
            name: float(model.predict(matrix)[0]) for name, model in self.others.items()
        }

    def comparison(self, task: str = "regression") -> pd.DataFrame:
        return compare(self.prepared, task)  # type: ignore[arg-type]

    def class_floor(self) -> float:
        data = split(self.prepared, "classification")
        counts = data.Y_test.value_counts(normalize=True)
        return float(counts.iloc[0]) if len(counts) else 0.0

    def cards(self) -> list[dict[str, Any]]:
        """One card per deployed model for the dashboard, with an honest metric name.

        A regressor has no accuracy. Labelling `r2` as "Model Accuracy" would put a number
        on the page that cannot exist, so the label comes from the task.
        """
        scores = self.artifact.scores
        return [
            _card(self.artifact.regressor_name, "R² Score", scores.get("regression", {}).get("r2")),
            _card(
                self.artifact.classifier_name,
                "Accuracy",
                scores.get("classification", {}).get("accuracy"),
            ),
        ]

    # -- writes ------------------------------------------------------------

    async def retrain(self, *, regressor: str, classifier: str, dataset: str) -> ModelService:
        """Fit and save a new pair off the event loop, and return the replacement."""
        return await asyncio.to_thread(
            self._retrain_blocking, regressor=regressor, classifier=classifier, dataset=dataset
        )

    def _retrain_blocking(
        self, *, regressor: str, classifier: str, dataset: str
    ) -> ModelService:
        prepared = self.prepared if dataset == self.prepared.name else prepare(dataset)
        artifact, _ = training.train(prepared, regressor=regressor, classifier=classifier)
        return ModelService(
            artifact=artifact, prepared=prepared, others=_fit_others(prepared)
        )


def _card(name: str, metric: str, value: float | None) -> dict[str, Any]:
    share = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    return {
        "name": name.replace("_", " ").title(),
        "metric": metric,
        "value": "n/a" if value is None else f"{value:,.4f}",
        "percent": round(share * 100),
    }


def _fit_others(prepared: Prepared) -> dict[str, Any]:
    """Fit every regressor once so the user page can show the spread between them."""
    data = split(prepared, "regression")
    fitted: dict[str, Any] = {}
    for name in COMPARISON_MODELS:
        model = build(name, task="regression")
        model.fit(data.X_train, data.Y_train)
        fitted[name] = model
    return fitted
