"""The one object the app keeps: the deployed models, and everything they need.

Built once at startup and held on `app.state`. Fitting takes seconds and the artefact is
tens of megabytes, so doing it per request would make every page load pay for it.

Retraining swaps the whole object rather than mutating it in place. A request that arrives
mid-retrain therefore sees either the old models or the new ones, never a half-swapped pair
where the regressor has been replaced and the thresholds have not.

Three collaborators, each holding one question, so this file stays about *answering a
request*: `catalogue.py` knows what can be asked about, `benchmarks.py` knows how every
model scores, and `presentation.py` with `verdict.py` know how to say it.

**The deployed dataset decides what the app offers.** A product artefact makes the ranked
shelf available and the described-shop page unanswerable; a store-day artefact does the
reverse. The surfaces read `has_products` and `supports_description` rather than checking
the dataset's name.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stockout import persistence, ranking
from stockout import train as training
from stockout.datasets import Prepared, prepare
from stockout.errors import StockoutError
from stockout.persistence import Artifact
from stockout.predict import Prediction, compare_row, predict
from stockout.ranking import Ranking

from .benchmarks import Benchmarks, cards, fit_every_regressor
from .catalogue import Catalogue, actual_for
from .presentation import StoreProfile, build_profiles
from .verdict import Verdict

DEFAULT_DATASET = "synthetic"


@dataclass
class ModelService:
    """The deployed artefact, the frame it came from, and the comparison fits."""

    artifact: Artifact
    prepared: Prepared
    others: dict[str, Any] = field(default_factory=dict)
    catalogue: Catalogue = field(init=False)
    benchmarks: Benchmarks = field(init=False)

    def __post_init__(self) -> None:
        self.catalogue = Catalogue(
            artifact=self.artifact,
            profiles=build_profiles(
                self.prepared.frame,
                self.artifact.thresholds,
                self.prepared.stores,
                has_products=self.artifact.has_products,
                unit=self.artifact.unit,
            ),
        )
        self.benchmarks = Benchmarks(prepared=self.prepared)

    @classmethod
    def load(cls, dataset: str = DEFAULT_DATASET) -> ModelService:
        """Read the saved artefact, or train one if this is a first run.

        The frame comes from the **artefact's** dataset, not from the argument: serving one
        source's models beside another source's frame would draw every "what it actually
        took" figure from the wrong file. `dataset` only decides what a first run fits.

        An artefact that exists but cannot be read is a refusal, not an invitation to fit a
        replacement — see the raise below.
        """
        if not persistence.exists():
            prepared = prepare(dataset)
            artifact, _ = training.train(prepared)
            return cls(artifact=artifact, prepared=prepared)

        try:
            artifact = persistence.load()
        except StockoutError as exc:
            # This used to catch every StockoutError and retrain on the default source, so
            # bumping ARTIFACT_VERSION silently replaced a deployment trained on real data
            # with one trained on the synthetic sample — same filename, same pages,
            # different dataset, nothing on screen saying so. `artifacts/` is not in
            # version control, so that loss is recoverable only by retraining.
            raise StockoutError(
                f"{exc} Refusing to fit a replacement automatically: the saved artefact "
                f"may have been trained on a different dataset, and overwriting it here "
                f"would lose that silently. Retrain deliberately with "
                f"`python -m stockout train --dataset <name>`."
            ) from exc

        prepared = prepare(artifact.dataset)
        return cls(artifact=artifact, prepared=prepared)

    # -- what can be asked -------------------------------------------------

    def store_options(self) -> list[StoreProfile]:
        return self.catalogue.entity_options()

    def item_options(self, store: str) -> list[StoreProfile]:
        return self.catalogue.item_options(store)

    def stores(self) -> list[str]:
        return self.catalogue.stores()

    def profile_for(self, entity: str) -> StoreProfile | None:
        return self.catalogue.profile_for(entity)

    def entity_for(self, store: str, item: str | None) -> str:
        return self.catalogue.entity_for(store, item)

    def categories(self) -> list[str]:
        return self.catalogue.categories()

    def entity_label(self) -> str:
        return self.artifact.entity_label

    def unit(self) -> str:
        return self.artifact.unit

    def has_products(self) -> bool:
        return self.artifact.has_products

    def supports_description(self) -> bool:
        """Whether the described-shop page can answer at all on this deployment.

        False where no cold-start model was fitted, which happens when the source
        publishes nothing describing an entity — see `train._fit_cold_start`.
        """
        return self.artifact.cold_regressor is not None

    def date_bounds(self) -> tuple[str, str]:
        """The span for which a true figure exists — the form's `min`, and its divider.

        No longer the form's `max`. The picker now opens past the end of the data, because
        a date beyond it is the only kind this tool answers that a lookup could not.
        """
        start, end = self.prepared.date_range()
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def actual_for(self, entity: str, date: str | pd.Timestamp) -> float | None:
        return actual_for(self.prepared, entity, date)

    # -- answers -----------------------------------------------------------

    def forecast(
        self,
        *,
        entity: str,
        date: str,
        promo: bool,
        state_holiday: str = "0",
        school_holiday: bool = False,
    ) -> Prediction:
        """Both answers for one entity-day, plus what every other regressor said."""
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

    def rank_products(
        self,
        *,
        store: str,
        date: str,
        promo: bool = False,
        state_holiday: str = "0",
        school_holiday: bool = False,
    ) -> Ranking:
        """Every item one shop stocks on one day, busiest first, scored in one pass."""
        return ranking.rank(
            self.artifact,
            store=store,
            date=date,
            promo=promo,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
            typical=self.catalogue.typical(),
        )

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
        """Every other regressor on the same row, in the headline's units.

        Delegated to `predict.compare_row` rather than predicting here. These models are
        fitted through the same de-trended split as the deployed one, so they answer in
        residuals; this method used to return those raw, putting a figure near zero beside
        a headline near five thousand. `Verdict.spread_low`/`spread_high` read this dict,
        so the range printed under the prediction was drawn from the wrong scale too.

        `compare_row` restores each store's line before returning, which is the same
        arithmetic the headline goes through — one implementation, so they cannot drift
        apart again.
        """
        return compare_row(
            self.artifact,
            self.others,
            entity=entity,
            date=date,
            promo=promo,
            state_holiday=state_holiday,
            school_holiday=school_holiday,
        )

    # -- warm-up -----------------------------------------------------------

    async def warm(self) -> None:
        """Fit the comparison regressors off the event loop, after the app is answering.

        These exist for one thing: the expandable list showing that seven regressors
        disagree about a row. Fitting them **before** serving cost thirteen minutes of
        startup on Rossmann — a random forest and a voting ensemble over 670,422 rows are
        most of it — during which the app answered nothing at all, for a feature nobody
        had asked for yet.

        Fitted on a sample would have been faster and dishonest: the page calls the spread
        "the honest width of the estimate", and a width widened by sampling noise is not
        that. So they are fitted in full, just not in the way of the door.

        Until this finishes `others` is empty, `Prediction.per_model` comes back empty, and
        the page simply omits the comparison section — which is the same thing it does for
        an artefact that never had one.
        """
        if self.others:
            return
        self.others = await asyncio.to_thread(fit_every_regressor, self.prepared)

    @property
    def is_warm(self) -> bool:
        """Whether the comparison regressors are fitted yet."""
        return bool(self.others)

    # -- the admin pages ---------------------------------------------------

    async def comparison_tables(self) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        return await self.benchmarks.tables()

    def cards(self) -> list[dict[str, Any]]:
        return cards(self.artifact)

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
        return ModelService(artifact=artifact, prepared=prepared)
