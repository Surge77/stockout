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

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import targets
from .config import RANDOM_SEED, TEST_SIZE, Task
from .data import schemas as s
from .datasets import Prepared, prepare
from .registry import build, model_names, spec_for
from .targets import DemandThresholds


@dataclass
class Split:
    """One held-out split, plus the thresholds fitted on its training rows."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    Y_train: pd.Series
    Y_test: pd.Series
    thresholds: DemandThresholds
    prepared: Prepared

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.X_train), len(self.X_test)


def split(
    prepared: Prepared,
    task: Task,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> Split:
    """Split the rows, then derive the classification labels from the training half."""
    frame = prepared.frame
    train_rows, test_rows = cast(
        "tuple[pd.DataFrame, pd.DataFrame]",
        train_test_split(frame, test_size=test_size, random_state=seed),
    )

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
    )


def score(task: Task, y_true: pd.Series, y_pred: Sequence[float]) -> dict[str, float]:
    """The metrics for one task, as a dict the results table turns into columns."""
    if task == "regression":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
        }
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def evaluate(name: str, task: Task, data: Split, *, seed: int = RANDOM_SEED) -> dict[str, object]:
    """Fit one registered model on the training rows and score it on the test rows."""
    model = build(name, task=task, seed=seed)
    started = time.perf_counter()
    model.fit(data.X_train, data.Y_train)
    fit_seconds = time.perf_counter() - started

    row: dict[str, object] = {"model": name}
    row.update(score(task, data.Y_test, model.predict(data.X_test)))
    row["fit_seconds"] = round(fit_seconds, 3)
    row["taught"] = "yes" if spec_for(name, task=task).taught else "no"
    return row


def compare(
    dataset: str | Prepared,
    task: Task,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Every registered model for `task`, on one split, in registry order."""
    prepared = prepare(dataset) if isinstance(dataset, str) else dataset
    data = split(prepared, task, test_size=test_size, seed=seed)
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
