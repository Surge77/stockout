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
from stockout.config import Task
from stockout.data.schemas import DATE as SCHEMA_DATE
from stockout.data.schemas import ENTITY as SCHEMA_ENTITY
from stockout.data.schemas import SALES as SCHEMA_SALES
from stockout.datasets import Prepared, prepare
from stockout.errors import StockoutError
from stockout.evaluate import compare, split
from stockout.hypothetical import (
    HypotheticalPrediction,
    StoreDescription,
    contributions,
    predict_described,
)
from stockout.persistence import Artifact
from stockout.predict import Prediction, predict, serving_row
from stockout.registry import build, model_names

from .presentation import StoreProfile, Verdict, build_profiles

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
    profiles: dict[str, StoreProfile] = field(default_factory=dict)

    #: Comparison tables, memoised per task. `compare` refits every registered model on
    #: the split, so an uncached `/admin/comparison` re-ran thirteen fits on every page
    #: load. The result depends only on `prepared` and the seed, both fixed for the life
    #: of this object, and a retrain builds a new object — so the cache cannot go stale.
    _tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _floor: float | None = field(default=None, repr=False)

    @classmethod
    def load(cls, dataset: str = DEFAULT_DATASET) -> ModelService:
        """Read the saved artefact, or train one if this is a first run."""
        prepared = prepare(dataset)
        try:
            artifact = persistence.load()
        except StockoutError:
            artifact, _ = training.train(prepared)
        return cls(
            artifact=artifact,
            prepared=prepared,
            others=_fit_others(prepared),
            profiles=build_profiles(prepared.frame, artifact.thresholds, prepared.stores),
        )

    # -- reads -------------------------------------------------------------

    def entities(self) -> list[str]:
        return self.artifact.entities

    def store_options(self) -> list[StoreProfile]:
        """Profiles in the dropdown's order: quietest store first, so the list reads
        as a range rather than as an arbitrary sequence of numbers."""
        return sorted(self.profiles.values(), key=lambda p: p.typical_sales)

    def profile_for(self, entity: str) -> StoreProfile | None:
        return self.profiles.get(str(entity))

    def entity_label(self) -> str:
        return self.artifact.entity_label

    def date_bounds(self) -> tuple[str, str]:
        """The span for which a true figure exists — the form's `min`, and its divider.

        No longer the form's `max`. The picker now opens past the end of the data, because
        a date beyond it is the only kind this tool answers that a lookup could not.
        """
        start, end = self.prepared.date_range()
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def actual_for(self, entity: str, date: str | pd.Timestamp) -> float | None:
        """What the store really took that day, where the dataset records it.

        This is the honest reply to the obvious objection: inside the fitted window a
        prediction is redundant, because the answer is already in the file. Rather than
        hide that, the panel shows both and lets the error stand next to the guess — which
        turns the redundancy from an embarrassment into the only place on the site where a
        visitor can watch the model be checked against something it cannot argue with.

        `None` for a day with no row: dates past the data, and closed days, which
        `features.drop_closed` removes before anything is fitted.
        """
        frame = self.prepared.frame
        stamp = pd.to_datetime(date)
        rows = frame[
            (frame[SCHEMA_ENTITY].astype("string") == str(entity))
            & (pd.to_datetime(frame[SCHEMA_DATE]) == stamp)
        ]
        if rows.empty:
            return None
        return float(rows[SCHEMA_SALES].iloc[0])

    def forecast(
        self,
        *,
        entity: str,
        date: str,
        promo: bool,
        state_holiday: str = "0",
        school_holiday: bool = False,
    ) -> Prediction:
        """Both answers for one store-day, plus what every other regressor said."""
        answer = predict(
            self.artifact,
            entity=entity,
            date=date,
            promo=promo,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
        )
        answer.per_model = self._every_model(
            entity=entity,
            date=date,
            promo=promo,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
        )
        return answer

    def describe(
        self,
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
        """Answer for a shop that does not exist, and say what each attribute was worth.

        Uses `artifact.cold_regressor`, not the deployed one — see `stockout.hypothetical`
        for why a model that knows store ids is the wrong tool for a store with no id.
        """
        description = StoreDescription(
            store_type=store_type,
            assortment=assortment,
            competition_distance=competition_distance,
            promo2=promo2,
        )
        answer = predict_described(
            self.artifact, description, date=date, promo=promo,
            state_holiday=state_holiday, school_holiday=school_holiday,
        )
        why = contributions(
            self.artifact, description, date=date, promo=promo,
            state_holiday=state_holiday, school_holiday=school_holiday,
        )
        return answer, why

    def verdict(self, answer: Prediction) -> Verdict | None:
        """The prediction, phrased for somebody deciding how much to order."""
        profile = self.profile_for(answer.entity)
        if profile is None:
            return None
        values = list(answer.per_model.values()) or [answer.sales]
        return Verdict(
            sales=answer.sales,
            demand_class=answer.demand_class,
            profile=profile,
            spread_low=min(values),
            spread_high=max(values),
        )

    def _every_model(
        self,
        *,
        entity: str,
        date: str,
        promo: bool,
        state_holiday: str,
        school_holiday: bool,
    ) -> dict[str, float]:
        """Every other regressor on the same row.

        `state_holiday` is threaded through because it was silently defaulting to `"0"`
        here: the comparison list was answering an ordinary day while the headline above
        it answered the holiday the visitor asked for, and the two were displayed as
        disagreeing models rather than as two different questions.
        """
        matrix = serving_row(
            self.artifact,
            entity=entity,
            date=date,
            promo=promo,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
        )
        return {
            name: float(model.predict(matrix)[0]) for name, model in self.others.items()
        }

    def comparison(self, task: Task = "regression") -> pd.DataFrame:
        """Every registered model scored on the split, fitted once and kept."""
        if task not in self._tables:
            self._tables[task] = compare(self.prepared, task)
        return self._tables[task]

    def class_floor(self) -> float:
        """The share of the largest class: what "always guess the commonest" would score.

        Cached with the tables. It re-derives the split, which is cheap, but it is asked
        for on the same page and there is no reason to do it twice.
        """
        if self._floor is None:
            data = split(self.prepared, "classification")
            counts = data.Y_test.value_counts(normalize=True)
            self._floor = float(counts.iloc[0]) if len(counts) else 0.0
        return self._floor

    async def comparison_tables(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """Both tables and the majority-class floor, computed off the event loop.

        `compare` fits every registered model for a task. Doing that inside an `async def`
        handler blocks the single event loop thread, so one admin opening this page froze
        every other request — including other people's forecasts — until it finished.
        """
        return await asyncio.to_thread(self._comparison_blocking)

    def _comparison_blocking(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        return self.comparison("regression"), self.comparison("classification"), self.class_floor()

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
            artifact=artifact,
            prepared=prepared,
            others=_fit_others(prepared),
            profiles=build_profiles(prepared.frame, artifact.thresholds, prepared.stores),
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
