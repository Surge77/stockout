"""Where every hyperparameter in this project comes from.

Without this module `alpha=1.0` in `linear.py` and `n_neighbors=15` in `kernels.py` are
magic numbers, and a viva question about either has no answer beyond "it is the default".
With it they are the defaults *until searched*, and the search is reproducible. ADR 0017.

Three decisions make the search trustworthy rather than decorative.

**The folds respect time.** `cv=TimeSeriesSplit`, never `KFold`. A grid search that
selects hyperparameters against shuffled folds picks the model that best predicts its own
past, and then the honest test window discovers otherwise. `gap` is passed through for the
same reason `features/lags.py` refuses a short lag: a fold whose training rows end the day
before its validation rows begin is tuning for a one-day horizon.

**The search runs on the same rows the model would fit on.** `unfitted_pipeline` and
`training_frame` come from the adapter, so a capped model is tuned on its cap and an
uncapped one on everything. Tuning on a different row set than the final fit chooses
parameters for a model nobody builds.

**`n_jobs=1` for the search, parallelism inside the estimator.** On Windows, joblib's loky
backend spawns processes and pickles the whole design matrix into each one. Four workers
on a 160 MB matrix is 640 MB of copies plus a re-import of sklearn per task, and it is
routinely *slower* than one process. The forests already set `n_jobs=-1` internally, where
threads share the array instead of copying it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from ..config import DEFAULT_HORIZON_DAYS, RANDOM_SEED
from ..errors import BacktestError
from ..split.strategies import assert_date_major
from .grids import candidate_count, grid_for
from .registry import build
from .spec import Task

#: Fewer folds than the backtest uses. A search costs candidates x folds fits, and the
#: point of cross-validating here is to avoid tuning against one accident rather than to
#: produce a publishable score — the honest number comes from the held-out window after.
DEFAULT_CV_FOLDS = 3

#: What each task is scored on during the search. R2 for regression because it is the
#: metric the comparison table leads with; macro-F1 for classification because plain
#: accuracy would let a search optimise away the smallest class and call it an improvement.
SCORING: dict[str, str] = {"regression": "r2", "classification": "f1_macro"}


@dataclass(frozen=True)
class SearchResult:
    """What one grid search found, and what it cost to find it."""

    model: str
    task: str
    candidates: int
    folds: int
    rows: int
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = float("nan")
    seconds: float = 0.0

    @property
    def searched(self) -> bool:
        """False when the model had nothing to tune, which is not an error."""
        return self.candidates > 0

    def as_row(self) -> dict[str, object]:
        return {
            "model": self.model,
            "task": self.task,
            "candidates": self.candidates,
            "fits": self.candidates * self.folds,
            "rows": self.rows,
            "best_score": self.best_score,
            "seconds": round(self.seconds, 1),
            "best_params": _render(self.best_params),
        }


def tune(
    frame: pd.DataFrame,
    name: str,
    *,
    task: Task,
    horizon: int = DEFAULT_HORIZON_DAYS,
    folds: int = DEFAULT_CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> SearchResult:
    """Search one model's grid against time-ordered folds. Returns what it found.

    Refuses a frame that is not date-major, because `TimeSeriesSplit` slices on row
    position and this package's default ordering is store-major — see
    `split/strategies.py`, which is the module that bug lives in the docstring of.
    """
    assert_date_major(frame)

    grid = grid_for(name, task=task)
    model = build(name, task=task, horizon=horizon, seed=seed)
    training = model.training_frame(frame)

    if not grid:
        return SearchResult(
            model=name, task=task, candidates=0, folds=folds, rows=len(training)
        )

    search = GridSearchCV(
        estimator=model.unfitted_pipeline(frame),
        param_grid=grid,
        scoring=SCORING[task],
        cv=TimeSeriesSplit(n_splits=folds, gap=horizon),
        n_jobs=1,
        refit=False,
        # Fail loudly. The default scores a crashed candidate as NaN and carries on, so a
        # pipeline that is broken for half the grid still reports a cheerful best_score
        # from the half that worked. sklearn's own stub types this as a float and accepts
        # the string at runtime, hence the cast.
        error_score=cast(float, "raise"),
    )

    started = time.perf_counter()
    search.fit(training, model.target_values(training))
    elapsed = time.perf_counter() - started

    return SearchResult(
        model=name,
        task=task,
        candidates=candidate_count(grid),
        folds=folds,
        rows=len(training),
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
        seconds=elapsed,
    )


def tune_many(
    frame: pd.DataFrame,
    names: list[str],
    *,
    task: Task,
    horizon: int = DEFAULT_HORIZON_DAYS,
    folds: int = DEFAULT_CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Search several models and return one row each, in the order asked for."""
    if not names:
        raise BacktestError("nothing to tune")
    results = [
        tune(frame, name, task=task, horizon=horizon, folds=folds, seed=seed)
        for name in names
    ]
    return pd.DataFrame([result.as_row() for result in results])


def _render(params: dict[str, Any]) -> str:
    """`estimate__alpha: 0.1` reads better in a table than the full pipeline path.

    The prefix is how sklearn addresses a step and is noise to a reader, who already
    knows which model the row is about.
    """
    if not params:
        return "-"
    trimmed = {key.split("__")[-1]: value for key, value in params.items()}
    return ", ".join(f"{key}={value}" for key, value in sorted(trimmed.items()))
