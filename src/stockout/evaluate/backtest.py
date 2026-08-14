"""The rolling-origin backtest loop.

Two choices here are load-bearing.

**A fresh model per fold.** `model_factory` is a callable, not an instance. Reusing one
fitted object across folds leaks the last fold's fit into the next one, and because the
scores only improve slightly it is very hard to notice.

**Closed days are excluded from every metric.** A shut store sells zero and every model
predicts zero, so including those rows adds a large block of perfect predictions — on
Rossmann roughly a seventh of all of them. That flatters `mae` and `rmse`, which are
per-row averages, by an amount bought entirely by predicting that a shut shop sells
nothing. Kaggle's own metric drops those rows; so does this.

Note the exception, because it is easy to assume otherwise: `wmape` is *unaffected*. A
closed day adds zero to its numerator and zero to its denominator, so it cancels exactly.
That robustness is part of why it is the primary metric here, and `test_backtest.py`
asserts both halves of the behaviour so neither can drift unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..config import DEFAULT_MIN_TRAIN_DAYS
from ..data import schemas as s
from ..models.baselines import SeasonalNaive
from ..split.rolling import rolling_origin, split_frame
from . import metrics

#: A zero-argument callable producing an unfitted forecaster. Typed loosely because a
#: `Forecaster` is a structural protocol and every baseline class satisfies it without
#: inheriting from anything.
ModelFactory = Callable[[], Any]

RESULT_COLUMNS: tuple[str, ...] = (
    "fold",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "n_train",
    "n_test",
    "mae",
    "rmse",
    "wmape",
    "rmspe",
    "mase",
)


def backtest(
    frame: pd.DataFrame,
    model_factory: ModelFactory,
    *,
    n_folds: int,
    horizon: int,
    gap: int = 0,
    min_train_days: int = DEFAULT_MIN_TRAIN_DAYS,
    expanding: bool = True,
    exclude_closed: bool = True,
) -> pd.DataFrame:
    """Fit and score one model across rolling-origin folds. One row per fold.

    `mase` is scaled by `SeasonalNaive` refitted on the same training window, so passing
    `SeasonalNaive` as the factory yields exactly 1.0 in every fold — a self-check the
    test suite asserts.
    """
    folds = rolling_origin(
        frame[s.DATE],
        n_folds=n_folds,
        horizon=horizon,
        gap=gap,
        min_train_days=min_train_days,
        expanding=expanding,
    )

    rows = []
    for fold in folds:
        train, test = split_frame(frame, fold)

        model = model_factory()
        model.fit(train)
        predicted = model.predict(test)

        baseline = SeasonalNaive().fit(train).predict(test)

        scored = test if not exclude_closed else test[test[s.OPEN] == 1]
        mask = scored.index
        actual = scored[s.SALES]

        rows.append(
            {
                "fold": fold.index,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train": len(train),
                "n_test": len(scored),
                "mae": metrics.mae(actual, predicted.loc[mask]),
                "rmse": metrics.rmse(actual, predicted.loc[mask]),
                "wmape": metrics.wmape(actual, predicted.loc[mask]),
                "rmspe": metrics.rmspe(actual, predicted.loc[mask]),
                "mase": metrics.mase(
                    actual, predicted.loc[mask], y_baseline=baseline.loc[mask]
                ),
            }
        )

    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def summarise(results: pd.DataFrame) -> pd.Series:
    """Mean of the numeric scores across folds.

    A mean, not a best. Reporting the best fold is how a backtest becomes a sales pitch.
    """
    numeric = ["mae", "rmse", "wmape", "rmspe", "mase"]
    summary = results[numeric].mean()
    summary["folds"] = float(len(results))
    return summary
