"""Score a fitted model, and run every registered model over one split.

Three metrics, all taught: `r2_score` and `mean_squared_error` for regression,
`accuracy_score` for classification. Nothing else is reported, because nothing else
appears in the course material.

`compare` is the point of the module. One `train_test_split`, then every model in the
registry fitted on the same training rows and scored on the same test rows — so the only
thing that differs between two rows of the table is the estimator.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import cast

import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TEST_SIZE, Task
from .datasets import Dataset, load, xy
from .registry import build, model_names, spec_for


def split(
    dataset: Dataset,
    task: Task,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the dataset and hold out `test_size` of it. The taught split, everywhere."""
    features, target = xy(load(dataset), dataset, task)
    # `train_test_split` is typed as returning a plain list of whatever it was handed,
    # so the frame/series pair it actually returns here has to be stated.
    return cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]",
        train_test_split(features, target, test_size=test_size, random_state=seed),
    )


def score(task: Task, y_true: pd.Series, y_pred: Sequence[float]) -> dict[str, float]:
    """The metrics for one task, as a dict the results table turns into columns."""
    if task == "regression":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
        }
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def evaluate(
    name: str,
    task: Task,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.Series,
    Y_test: pd.Series,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, float | str]:
    """Fit one registered model on the training rows and score it on the test rows."""
    model = build(name, task=task, seed=seed)
    started = time.perf_counter()
    model.fit(X_train, Y_train)
    fit_seconds = time.perf_counter() - started

    row: dict[str, float | str] = {"model": name}
    row.update(score(task, Y_test, model.predict(X_test)))
    row["fit_seconds"] = round(fit_seconds, 3)
    row["taught"] = "yes" if spec_for(name, task=task).taught else "no"
    return row


def compare(
    dataset: Dataset,
    task: Task,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Every registered model for `task`, on one split of `dataset`, in registry order."""
    X_train, X_test, Y_train, Y_test = split(dataset, task, test_size=test_size, seed=seed)
    rows = [
        evaluate(name, task, X_train, X_test, Y_train, Y_test, seed=seed)
        for name in model_names(task)
    ]
    return pd.DataFrame(rows)


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
