"""Run every registered model against one held-out window and tabulate what happened.

This is the table the notebooks chart, `stockout compare` prints, the web app's admin page
displays and `docs/results.md` quotes, so it is built once here rather than four times in
four places.

**The thresholds are refitted on the training slice.** `dataset.prepare` labels the whole
frame, which is right for looking at data and wrong for scoring a model: a tercile cut
point is a statistic, and one computed over the test window has let the test window vote
on its own labels. Every run below refits them on the training rows alone. It is a small
function call and it is the difference between a number and a number you can defend.

**Wall clock is a reported column, not a footnote.** A model that scores 0.958 in two
seconds and one that scores 0.961 in forty minutes are not the same result, and a table
that omits the second axis is quietly recommending the wrong one.

**Rows fitted is also a column**, because the kernel models are capped. Reading `svr`'s
score without seeing that it saw 5,000 rows while `ridge` saw all of them is reading two
numbers as though they were comparable. ADR 0015.
"""

from __future__ import annotations

import time

import pandas as pd

from ..config import RANDOM_SEED
from ..data import schemas as s
from ..models.registry import build, model_names, spec_for
from ..models.spec import Task
from ..split.strategies import time_holdout
from ..targets import DEMAND_CLASS_CODE, add_demand_class, fit_thresholds
from . import metrics
from .classification import score as score_classes

REGRESSION_COLUMNS: tuple[str, ...] = (
    "model", "train_rows", "test_rows", "r2", "wmape", "mae", "rmse", "seconds", "note",
)  # fmt: skip

CLASSIFICATION_COLUMNS: tuple[str, ...] = (
    "model", "train_rows", "test_rows", "accuracy", "macro_f1", "adjacent",
    "recall_low", "recall_medium", "recall_high", "seconds", "note",
)  # fmt: skip


def compare(
    frame: pd.DataFrame,
    *,
    task: Task,
    test_days: int,
    gap_days: int,
    models: list[str] | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """One row per model, registry order — simplest first.

    Registry order rather than sorted by score. A table sorted by score answers "which
    won"; this order also answers "did the extra complexity pay", which is the question
    worth asking and the harder one to fake.
    """
    train, test = time_holdout(frame, test_days=test_days, gap_days=gap_days)
    train, test = _relabel(train, test)

    chosen = models or list(model_names(task))
    rows = [_score_one(name, train, test, task=task, seed=seed) for name in chosen]

    columns = REGRESSION_COLUMNS if task == "regression" else CLASSIFICATION_COLUMNS
    return pd.DataFrame(rows, columns=list(columns))


def _relabel(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the demand-class cut points on the training rows, then apply to both.

    `dataset.prepare` fitted them over everything, which is fine for exploration and is
    leakage here. Doing it again costs one pass over a column and removes the objection.
    """
    thresholds = fit_thresholds(train)
    return add_demand_class(train, thresholds), add_demand_class(test, thresholds)


def _score_one(
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    task: Task,
    seed: int,
) -> dict[str, object]:
    model = build(name, task=task, seed=seed)

    started = time.perf_counter()
    model.fit(train)
    predicted = model.predict(test)
    elapsed = time.perf_counter() - started

    common: dict[str, object] = {
        "model": name,
        # Two row counts, not one. They used to collide under a single "rows" key and
        # the scoring half won, which quietly erased the fact that svr saw 5,000
        # training rows while ridge saw every one. That erasure is exactly what
        # ADR 0015 says the table must not do.
        "train_rows": model.training_rows,
        "seconds": round(elapsed, 1),
        "note": spec_for(name, task=task).note,
    }
    scored = (
        _regression_scores(test, predicted)
        if task == "regression"
        else score_classes(test[DEMAND_CLASS_CODE], predicted).as_row()
    )
    return {**common, **scored}


def _regression_scores(test: pd.DataFrame, predicted: pd.Series) -> dict[str, object]:
    """Scored on trading rows only.

    A closed store sells zero and every model predicts it, so including those rows adds a
    block of perfect predictions that flatters `mae` and `rmse` — both per-row averages —
    by an amount bought entirely by predicting that a shut shop sells nothing. `wmape` is
    unaffected either way, since a closed day contributes zero to both its numerator and
    its denominator, but the two are reported side by side and must mean the same thing.
    """
    trading = test[test[s.OPEN] == 1]
    actual = trading[s.SALES]
    fitted = predicted.loc[trading.index]
    return {
        "test_rows": len(trading),
        "r2": metrics.r2(actual, fitted),
        "wmape": metrics.wmape(actual, fitted),
        "mae": metrics.mae(actual, fitted),
        "rmse": metrics.rmse(actual, fitted),
    }


def to_markdown(table: pd.DataFrame, *, task: Task) -> str:
    """The comparison as a markdown table, with the winner named underneath.

    Naming the winner is the point. A table that leaves the reader to scan a column and
    work it out is a table that will be misread, and the metric that decides is not the
    one most readers would pick — WMAPE for regression, macro-F1 for classification.
    """
    if table.empty:
        return "**Nothing was compared.**"

    # Lower is better for WMAPE, higher for macro-F1 — the two halves disagree on which
    # direction "best" points, and hard-coding either one silently names the loser.
    metric = "wmape" if task == "regression" else "macro_f1"
    scores = table[metric].to_numpy(dtype="float64")
    position = int(scores.argmin() if task == "regression" else scores.argmax())

    # `.to_dict()` rather than `.loc[...]`, which a type checker cannot narrow from a
    # row-or-frame accessor down to scalars.
    best = table.iloc[position].to_dict()
    best_metric = float(best[metric])
    best_rows = int(best["train_rows"])

    header = f"| {' | '.join(table.columns)} |"
    rule = f"|{'---|' * len(table.columns)}"
    body = "\n".join(
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in table.itertuples(index=False)
    )
    verdict = (
        f"\n\n**`{best['model']}` wins on {metric}** at {best_metric:.3f}, "
        f"fitted on {best_rows:,} training rows in {best['seconds']}s."
    )
    return f"{header}\n{rule}\n{body}{verdict}"


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
