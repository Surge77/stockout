"""The one seam between the web app and the forecasting package.

Every call into `stockout` goes through here, so that the routers stay about HTTP and this
file stays about models. Three things it owns, each of which is a performance decision the
routers must not be allowed to make by accident:

**The artifact is loaded once, at startup.** A fitted pipeline is tens of megabytes of
numpy arrays; `joblib.load` on every request would dominate the response time and do it
invisibly, because each individual load is only a second.

**The history frame is prepared once and cached.** `predict.forecast` needs a store's own
past to rebuild its lags, and `dataset.prepare` reads two CSVs, joins them, builds the
calendar and lag columns and drops the warm-up rows. Doing that per request would be
seconds of identical work.

**Fitting never runs on the event loop.** Training and the model comparison are seconds of
CPU with no await in them, so they go through `run_in_executor`. Left on the loop they
would block every other request in the process, including the health check.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from stockout import persistence
from stockout import train as training
from stockout.data import schemas as s
from stockout.dataset import prepare
from stockout.errors import StockoutError
from stockout.evaluate.comparison import compare
from stockout.models.registry import model_names
from stockout.predict import Forecast, forecast

Task = Literal["regression", "classification"]

#: The held-out window `train` scores on before refitting on everything. Four weeks, the
#: same default the CLI uses, so a number on the admin page matches a number from the
#: command line rather than nearly matching it.
TEST_DAYS = 28


@dataclass
class ModelState:
    """What the app knows about the deployed model right now."""

    artifact: persistence.Artifact | None = None
    history: pd.DataFrame | None = None
    stores: list[int] = field(default_factory=list)
    last_date: dt.date | None = None
    error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.artifact is not None and self.history is not None

    @property
    def servable_until(self) -> dt.date | None:
        """The furthest day a forecast can honestly reach.

        The model reads `sales_lag_{horizon}`, so to predict day D it needs the actual
        sales of `D - horizon`. Past `last_date + horizon` the lag would have to be a
        prediction of a prediction. `predict` refuses beyond this; the form uses it to
        stop somebody asking in the first place.
        """
        if self.artifact is None or self.last_date is None:
            return None
        return self.last_date + pd.Timedelta(days=self.artifact.horizon)


class ModelService:
    """Loads the artifact, answers forecasts, and runs the two slow admin jobs."""

    def __init__(self) -> None:
        self.state = ModelState()
        #: One trainer at a time. Two concurrent fits would race to write the same
        #: artifact file and the loser would be silently discarded.
        self._training = asyncio.Lock()

    # --- loading ---------------------------------------------------------------------

    def load(self) -> ModelState:
        """Read the artifact and prepare the history. Records failure rather than raising.

        A missing artifact is the *normal* first-run state — nobody has trained yet — so
        it becomes a message on the page rather than a stack trace at startup. The app
        must come up even with no model, or an admin can never log in to make one.
        """
        try:
            artifact = persistence.load()
            prepared = prepare(horizon=artifact.horizon)
        except (StockoutError, FileNotFoundError, OSError) as exc:
            self.state = ModelState(error=str(exc))
            return self.state

        frame = prepared.frame
        self.state = ModelState(
            artifact=artifact,
            history=frame,
            stores=sorted(int(store) for store in frame[s.STORE].unique()),
            last_date=frame[s.DATE].max().date(),
        )
        return self.state

    # --- the user-facing question ----------------------------------------------------

    def predict(
        self,
        *,
        store: int,
        when: dt.date,
        promo: bool = False,
        school_holiday: bool = False,
        is_open: bool = True,
    ) -> Forecast:
        """Both halves of the answer for one store-day: the number and the class.

        Regression and classification come back together because they are the same
        question asked twice — *how much* and *how busy* — and a page that showed one
        without the other would be hiding half the model.
        """
        if self.state.artifact is None or self.state.history is None:
            raise StockoutError("no model has been trained yet")
        if store not in self.state.stores:
            raise StockoutError(f"store {store} is not in the training data")

        return forecast(
            self.state.artifact,
            self.state.history,
            store=store,
            date=pd.Timestamp(when),
            promo=int(promo),
            school_holiday=int(school_holiday),
            is_open=int(is_open),
        )

    # --- the two slow admin jobs -----------------------------------------------------

    async def retrain(self, *, regressor: str, classifier: str) -> persistence.Artifact:
        """Fit both tasks on everything, save, and reload what the app serves.

        Held behind a lock and pushed off the event loop. The reload at the end is the
        point: without it the admin sees "training complete" while every user keeps
        being served the previous model.
        """
        _require_known(regressor, task="regression")
        _require_known(classifier, task="classification")

        async with self._training:
            artifact = await _off_the_loop(self._fit, regressor, classifier)
            self.load()
            return artifact

    def _fit(self, regressor: str, classifier: str) -> persistence.Artifact:
        prepared = prepare()
        artifact = training.train(
            prepared.frame,
            horizon=prepared.horizon,
            regressor=regressor,
            classifier=classifier,
            test_days=TEST_DAYS,
        )
        persistence.save(artifact)
        return artifact

    async def comparison(self, task: Task, models: list[str] | None = None) -> pd.DataFrame:
        """Every registered model on one holdout — the table the admin page shows."""
        return await _off_the_loop(self._compare, task, models)

    def _compare(self, task: Task, models: list[str] | None) -> pd.DataFrame:
        for name in models or []:
            _require_known(name, task=task)
        prepared = prepare()
        return compare(
            prepared.frame,
            task=task,
            test_days=TEST_DAYS,
            gap_days=prepared.horizon,
            models=models,
        )


async def _off_the_loop(function: Any, *args: Any) -> Any:
    """Run a CPU-bound call in the default executor.

    `train` and `compare` fit scikit-learn models: seconds of pure computation with no
    await anywhere inside. On the event loop they would block every other request in the
    process until they finished.
    """
    return await asyncio.get_running_loop().run_in_executor(None, function, *args)


def _require_known(name: str, *, task: Task) -> None:
    """Reject an unregistered model before anything is loaded.

    A model name arriving from a form is untrusted input, and the registry is the only
    list allowed to say which ones exist.
    """
    if name not in model_names(task):
        raise StockoutError(f"unknown {task} model {name!r}")
