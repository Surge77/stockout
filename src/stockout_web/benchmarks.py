"""How every registered model scores on the deployed split, and the dashboard's cards.

Separate from `service.py` because it answers a different question. That module holds *the
models this app serves*; this one holds *how they compare with the ones it does not* — the
admin-facing evidence for why the deployed pair is the deployed pair.

Everything here is memoised. `evaluate.compare` refits every registered model for a task,
so an uncached comparison page re-ran thirteen fits on every load. The results depend only
on the prepared frame and the seed, both fixed for the life of one of these objects, and a
retrain builds a new one — so the cache cannot go stale.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stockout.config import Task
from stockout.datasets import Prepared
from stockout.evaluate import compare, split
from stockout.persistence import Artifact
from stockout.registry import build, model_names


@dataclass
class Benchmarks:
    """The comparison tables for one prepared frame, computed on demand and kept."""

    prepared: Prepared
    _tables: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _floor: float | None = field(default=None, repr=False)

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

    async def tables(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """Both tables and the majority-class floor, computed off the event loop.

        `compare` fits every registered model for a task. Doing that inside an `async def`
        handler blocks the single event loop thread, so one admin opening this page froze
        every other request — including other people's forecasts — until it finished.
        """
        return await asyncio.to_thread(self._blocking)

    def _blocking(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        return self.comparison("regression"), self.comparison("classification"), self.class_floor()


def cards(artifact: Artifact) -> list[dict[str, Any]]:
    """One card per deployed model for the dashboard, with an honest metric name.

    A regressor has no accuracy. Labelling `r2` as "Model Accuracy" would put a number on
    the page that cannot exist, so the label comes from the task.
    """
    scores = artifact.scores
    return [
        _card(artifact.regressor_name, "R² Score", scores.get("regression", {}).get("r2")),
        _card(
            artifact.classifier_name,
            "Accuracy",
            scores.get("classification", {}).get("accuracy"),
        ),
    ]


def _card(name: str, metric: str, value: float | None) -> dict[str, Any]:
    share = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    return {
        "name": name.replace("_", " ").title(),
        "metric": metric,
        "value": "n/a" if value is None else f"{value:,.4f}",
        "percent": round(share * 100),
    }


#: Fitting all seven regressors so the forecast page can show that they disagree costs
#: about twenty seconds. Done once at startup and on retrain, never per request.
COMPARISON_MODELS: tuple[str, ...] = model_names("regression")


def fit_every_regressor(prepared: Prepared) -> dict[str, Any]:
    """Fit each registered regressor once, so a page can show the spread between them.

    Fitted through the same de-trended split as the deployed model, so every one of these
    answers in residuals — `predict.compare_row` is what puts the line back before any of
    them reaches a page.
    """
    data = split(prepared, "regression")
    fitted: dict[str, Any] = {}
    for name in COMPARISON_MODELS:
        model = build(name, task="regression")
        model.fit(data.X_train, data.Y_train)
        fitted[name] = model
    return fitted
