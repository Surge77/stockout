"""Score a fitted model, and run every registered model over one split.

Three metrics, all taught: `r2_score` and `mean_squared_error` for regression,
`accuracy_score` for classification.

`compare` is the point of the module. One `train_test_split`, then every model in the
registry fitted on the same training rows and scored on the same test rows — so the only
thing that differs between two rows of the table is the estimator.

**The demand thresholds are fitted inside the split, never outside it.** A tercile cut
point is a statistic; one computed over the whole frame lets the test rows vote on their
own labels. `split` does the split first and derives the labels second, which is a two-line
difference and the difference between a number and a number that can be defended.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import targets, trend
from .config import DEFAULT_SPLIT, DETREND, RANDOM_SEED, TEST_SIZE, SplitStrategy, Task
from .data import schemas as s
from .datasets import Prepared, prepare
from .errors import StockoutError
from .registry import build, model_names, spec_for
from .targets import DemandThresholds
from .trend import StoreTrends


@dataclass
class Split:
    """One held-out split, plus everything fitted on its training rows alone."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    Y_train: pd.Series
    Y_test: pd.Series
    thresholds: DemandThresholds
    prepared: Prepared

    #: Per-store lines fitted on the training rows, or `None` when de-trending is off.
    #: When set, `Y_train` and `Y_test` hold *residuals* for regression rather than sales,
    #: and `evaluate` puts the line back before scoring — so every r² in a results table
    #: is still measured against real sales and stays comparable with an undetrended run.
    trends: StoreTrends | None = None

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.X_train), len(self.X_test)

    def rows_for(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """The original frame rows behind a feature matrix, by index.

        Re-trending needs each row's entity and date, and the matrix carries neither —
        it holds `entity_code`, which is an encoding, not an id.
        """
        return self.prepared.frame.loc[matrix.index]


def split(
    prepared: Prepared,
    task: Task,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    strategy: SplitStrategy = DEFAULT_SPLIT,
    detrend: bool = DETREND,
) -> Split:
    """Split the rows, then derive the targets from the training half alone.

    Two strategies, and the choice changes what the score means.

    `"chronological"` holds out the last `test_size` of the *calendar*: the model is fitted
    on the early period and scored on a later one it has never seen. This is the default
    because it is the only arrangement that matches how the thing would be used.

    `"shuffled"` is `train_test_split`, as taught, and it is kept for exactly that reason —
    plus it is the control arm `leakage.py` needs. It is not a safe default here. No
    feature in this package is a lag, so the obvious leak is absent, but the calendar
    features are enough on their own: `entity_code` with `month`, `day_of_month` and
    `day_of_week` very nearly identifies a single row, so a deep tree can land a test row
    in a leaf built from the days either side of it — whose residuals are correlated by
    construction. `leakage.measure` puts a number on what that is worth.
    """
    frame = prepared.frame
    if strategy == "shuffled":
        train_rows, test_rows = cast(
            "tuple[pd.DataFrame, pd.DataFrame]",
            train_test_split(frame, test_size=test_size, random_state=seed),
        )
    else:
        train_rows, test_rows = _by_calendar(frame, test_size=test_size)

    # The trend, like the thresholds, is a statistic fitted on the training rows alone.
    # One measured over the whole frame lets the held-out period set the line it is then
    # scored against, which is the same mistake as global terciles wearing a different hat.
    trends = trend.fit(train_rows) if detrend else None
    if trends is not None:
        train_rows = _detrended(train_rows, trends)
        test_rows = _detrended(test_rows, trends)

    thresholds = targets.fit(train_rows)
    matrix = prepared.feature_columns

    if task == "regression":
        y_train = train_rows[s.SALES]
        y_test = test_rows[s.SALES]
    else:
        y_train = targets.apply(train_rows, thresholds)
        y_test = targets.apply(test_rows, thresholds)

    return Split(
        X_train=train_rows[matrix],
        X_test=test_rows[matrix],
        Y_train=y_train,
        Y_test=y_test,
        thresholds=thresholds,
        prepared=prepared,
        trends=trends,
    )


def _detrended(rows: pd.DataFrame, trends: StoreTrends) -> pd.DataFrame:
    """The same rows with `sales` replaced by its distance from the store's own line.

    Both targets are built from this column, so de-trending it once here fixes both at
    the same time. Regression then predicts *how unusual was this day*, and the demand
    terciles are cut on the same quantity — which is what stops the classes drifting as
    stores grow. On Rossmann the untouched labels reach 40.6% High in the held-out period
    against 33.3% in training, purely because the chain gained about 5% a year.
    """
    out = rows.copy()
    out[s.SALES] = trend.remove(rows, trends)
    return out


def _by_calendar(frame: pd.DataFrame, *, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_size` of the calendar.

    Cut on the date, not on a row position. A store-day is one row per store, so slicing
    the last fifth of the *rows* would put some stores' copies of a date in training and
    others' in test — the same day on both sides of the split, which is the leak this is
    supposed to remove.

    The boundary date itself goes to the test side, so every training row is strictly
    earlier than every test row.
    """
    dates = pd.to_datetime(frame[s.DATE])
    calendar = pd.Index(dates.unique()).sort_values()
    if len(calendar) < 2:
        raise StockoutError("a chronological split needs at least two distinct dates")

    cut_at = max(1, round(len(calendar) * (1.0 - test_size)))
    boundary = calendar[min(cut_at, len(calendar) - 1)]

    train_rows = frame[dates < boundary]
    test_rows = frame[dates >= boundary]
    if train_rows.empty or test_rows.empty:  # pragma: no cover - guarded by the length check
        raise StockoutError("a chronological split produced an empty side")
    return train_rows, test_rows


def score(
    task: Task, y_true: pd.Series, y_pred: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """The metrics for one task, as a dict the results table turns into columns."""
    if task == "regression":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
        }
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def training_rows(
    name: str, task: Task, data: Split, *, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """The rows one model is fitted on: the whole training half, or a sample of it.

    Only `svc` caps — see `ModelSpec.max_training_rows` for why quadratic training cost
    makes it necessary and what it costs in comparability. The sample is drawn with the
    run's seed, so a capped row is as reproducible as an uncapped one.

    Every model is scored on the *full* test half either way. The cap changes how much a
    model learnt from, never what it is judged against.
    """
    cap = spec_for(name, task=task).max_training_rows
    if cap is None or len(data.X_train) <= cap:
        return data.X_train, data.Y_train

    sample = data.X_train.sample(n=cap, random_state=seed)
    return sample, data.Y_train.loc[sample.index]


def scored_against(
    model: object, task: Task, data: Split
) -> tuple[pd.Series, Sequence[float] | np.ndarray]:
    """The truth and the prediction for the test half, both in real-sales units.

    On a de-trended split the model answers in residuals, so both sides have to be put
    back on the store's line before scoring. Skipping this would report how well the
    *residual* is predicted — an easier question with a flattering answer, and one not
    comparable with an undetrended run. `train` scores through here for the same reason.
    """
    predicted = model.predict(data.X_test)  # type: ignore[attr-defined]
    if task != "regression" or data.trends is None:
        return data.Y_test, predicted

    rows = data.rows_for(data.X_test)
    restored = trend.restore(predicted, rows[s.ENTITY], rows[s.DATE], data.trends)
    return rows[s.SALES], restored


def evaluate(name: str, task: Task, data: Split, *, seed: int = RANDOM_SEED) -> dict[str, object]:
    """Fit one registered model on the training rows and score it on the test rows."""
    model = build(name, task=task, seed=seed)
    X_fit, y_fit = training_rows(name, task, data, seed=seed)

    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started

    truth, predicted = scored_against(model, task, data)

    row: dict[str, object] = {"model": name}
    row.update(score(task, truth, predicted))
    row["fit_seconds"] = round(fit_seconds, 3)
    # Reported for every model, not only the capped one. A single asterisk on one row is
    # easy to miss; a column that is identical everywhere except one place is not.
    row["n_fit"] = len(X_fit)
    row["taught"] = "yes" if spec_for(name, task=task).taught else "no"
    return row


def compare(
    dataset: str | Prepared,
    task: Task,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    strategy: SplitStrategy = DEFAULT_SPLIT,
    detrend: bool = DETREND,
) -> pd.DataFrame:
    """Every registered model for `task`, on one split, in registry order."""
    prepared = prepare(dataset) if isinstance(dataset, str) else dataset
    data = split(
        prepared, task, test_size=test_size, seed=seed, strategy=strategy, detrend=detrend
    )
    rows = [evaluate(name, task, data, seed=seed) for name in model_names(task)]
    return pd.DataFrame(rows)


def baseline_accuracy(labels: pd.Series) -> float:
    """The share of the commonest class — what a classifier must beat to have learnt
    anything. Per-entity terciles put this near a third by construction."""
    if labels.empty:
        return 0.0
    return float(labels.value_counts(normalize=True).iloc[0])


def to_markdown(frame: pd.DataFrame, *, places: int = 4) -> str:
    """Any results frame as a markdown table, floats rounded, everything else left alone."""
    if frame.empty:
        return "_(no rows)_"

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda v, p=places: f"{v:,.{p}f}")

    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False)]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]

    def line(cells: Sequence[str]) -> str:
        padded = [cell.ljust(width) for cell, width in zip(cells, widths, strict=True)]
        return "| " + " | ".join(padded) + " |"

    rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), rule, *(line(row) for row in rows)])
