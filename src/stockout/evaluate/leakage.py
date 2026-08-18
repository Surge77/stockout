"""Four ways to be wrong about your own error, priced against one fixed future.

The obvious experiment — score a random split, score a time split, compare — does not
work, and it is worth saying why before showing what replaces it. Those two protocols
differ in training size, in test period *and* in leakage all at once, so the gap between
them cannot be attributed to any one of the three. An examiner who notices has ended the
discussion.

So the test window is **held fixed**. Every arm below is scored on the same final stretch
of the calendar, which no arm trains on. What varies is only how each arm selects its
training and validation rows out of everything before that window, and therefore only
what each arm *believes* about its own accuracy.

That reframes the claim from "this model is better" to "this protocol's estimate of its
error was wrong by this much", which is both the true statement and the more useful one:

    The random split told me 0.94. On the window I actually have to forecast, that same
    model scored 0.81. The random split's estimate of my error was off by 0.13.

Each arm reports three numbers. `internal` is what the protocol would have told you.
`future` is what the model then did on the held-out window. `optimism` is the difference,
and it is the column the whole module exists to produce. A protocol with an optimism near
zero is honest about itself even if its absolute score is mediocre; a protocol with a
large optimism is lying, and the size of the lie is the finding.

One model throughout — Ridge — because the comparison is between protocols and changing
the estimator as well would confound it again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ..config import RANDOM_SEED
from ..data import schemas as s
from ..features.build import feature_columns
from ..features.preprocess import column_roles, make_preprocessor
from ..models.adapter import to_float32
from ..split.strategies import random_split, time_holdout
from . import metrics

#: The column Rossmann ships, that predicts sales at about r = 0.9, and that nobody knows
#: six weeks ahead. `features/build.py::FEATURE_DENYLIST` refuses it, so the arm that
#: demonstrates the leak has to rename it to get past the guard — which is itself the
#: neatest evidence that the guard works.
SMUGGLED_CUSTOMERS = "customers_known_in_advance"

ARM_COLUMNS: tuple[str, ...] = ("arm", "protocol", "features", "internal", "future", "optimism")


@dataclass(frozen=True)
class Arm:
    """One protocol's honesty about itself."""

    arm: str
    protocol: str
    features: str
    internal: float
    future: float

    @property
    def optimism(self) -> float:
        """How much the protocol over-reported. Positive means it flattered itself."""
        return self.internal - self.future

    def as_row(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "protocol": self.protocol,
            "features": self.features,
            "internal": self.internal,
            "future": self.future,
            "optimism": self.optimism,
        }


def leakage_arms(
    frame: pd.DataFrame,
    *,
    test_days: int,
    gap_days: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run all four arms against one held-out future window. One row per arm."""
    history, future = time_holdout(frame, test_days=test_days, gap_days=gap_days)
    arms = [
        _honest(history, future, seed=seed),
        _shuffled(history, future, seed=seed),
        _preprocessing_leak(history, future, seed=seed),
        _future_feature(history, future, seed=seed),
    ]
    return pd.DataFrame([arm.as_row() for arm in arms], columns=list(ARM_COLUMNS))


def _honest(history: pd.DataFrame, future: pd.DataFrame, *, seed: int) -> Arm:
    """Time-ordered validation, honest features, preprocessing inside the pipeline.

    The control. Its optimism is not expected to be zero — the validation window and the
    future window are different stretches of calendar and demand moves — but it is the
    only arm whose remaining optimism is honest ignorance rather than self-deception.
    """
    train, validate = time_holdout(history, test_days=len(future[s.DATE].unique()), gap_days=0)
    model = _fit(train, seed=seed)
    return Arm(
        arm="honest",
        protocol="time-ordered",
        features="denylist enforced",
        internal=_score(model, validate),
        future=_score(model, future),
    )


def _shuffled(history: pd.DataFrame, future: pd.DataFrame, *, seed: int) -> Arm:
    """`train_test_split(shuffle=True)` — the line in all fourteen course notebooks.

    Tuesday trains and the Wednesday either side of it validates. They share a store, a
    promotion, a season and nearly every lag value, so the validation score is closer to
    a recall test than a forecast.
    """
    train, validate = random_split(history, seed=seed)
    model = _fit(train, seed=seed)
    return Arm(
        arm="shuffled split",
        protocol="random",
        features="denylist enforced",
        internal=_score(model, validate),
        future=_score(model, future),
    )


def _preprocessing_leak(history: pd.DataFrame, future: pd.DataFrame, *, seed: int) -> Arm:
    """The scaler and imputer fitted on everything, before any split was made.

    The commonest leak in tutorial code, and the direct answer to "why does `Pipeline`
    exist". Nothing here looks like cheating: the model still never sees a validation
    *row*. It sees the validation set's mean, its variance and its median, which is
    enough to move the score and is exactly what fitting inside a fold prevents.
    """
    train, validate = time_holdout(history, test_days=len(future[s.DATE].unique()), gap_days=0)

    everything = pd.concat([history, future], ignore_index=True)
    roles = column_roles(everything, feature_columns=_candidates(everything))
    leaked = make_preprocessor(roles).fit(_trading(everything))

    estimator = Ridge(alpha=1.0)
    trading = _trading(train)
    estimator.fit(
        to_float32(np.asarray(leaked.transform(trading))),
        trading[s.SALES].to_numpy(dtype="float64"),
    )

    def score(rows: pd.DataFrame) -> float:
        scored = _trading(rows)
        predicted = estimator.predict(to_float32(np.asarray(leaked.transform(scored))))
        return metrics.r2(scored[s.SALES], np.clip(predicted, 0.0, None))

    return Arm(
        arm="preprocessing leak",
        protocol="time-ordered",
        features="scaler and imputer fitted on all rows",
        internal=score(validate),
        future=score(future),
    )


def _future_feature(history: pd.DataFrame, future: pd.DataFrame, *, seed: int) -> Arm:
    """`customers` allowed back in, under an assumed name.

    It is in `train.csv`, it correlates with `sales` at about 0.9, and on the day you
    forecast it does not exist. This arm scores beautifully on both windows and is
    undeployable for a single day, which is the point: a high score on a held-out set is
    not evidence that a model can be used.
    """
    smuggled_history = _smuggle(history)
    smuggled_future = _smuggle(future)
    train, validate = time_holdout(
        smuggled_history, test_days=len(future[s.DATE].unique()), gap_days=0
    )
    model = _fit(train, seed=seed)
    return Arm(
        arm="future feature",
        protocol="time-ordered",
        features=f"{s.CUSTOMERS} smuggled past the denylist",
        internal=_score(model, validate),
        future=_score(model, smuggled_future),
    )


def _smuggle(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename `customers` so `FEATURE_DENYLIST` stops recognising it.

    Deliberately awkward. The guard cannot be switched off from outside, so demonstrating
    what it prevents requires lying to it about a column name — and needing to do that is
    the strongest evidence available that the guard is load-bearing rather than decorative.
    """
    out = frame.copy()
    if s.CUSTOMERS in out.columns:
        out[SMUGGLED_CUSTOMERS] = out[s.CUSTOMERS].astype("float64")
    return out


def _candidates(frame: pd.DataFrame) -> list[str]:
    return [column for column in feature_columns(frame) if column != s.SALES]


def _trading(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame[s.OPEN] == 1] if s.OPEN in frame.columns else frame


def _fit(train: pd.DataFrame, *, seed: int) -> Pipeline:
    """One Ridge behind the shared preprocessor, fitted the correct way."""
    trading = _trading(train)
    roles = column_roles(trading, feature_columns=_candidates(trading))
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(roles)),
            ("compact", _compactor()),
            ("estimate", Ridge(alpha=1.0, random_state=seed)),
        ]
    )
    pipeline.fit(trading, trading[s.SALES].to_numpy(dtype="float64"))
    return pipeline


def _compactor() -> object:
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(to_float32, feature_names_out="one-to-one")


def _score(pipeline: Pipeline, rows: pd.DataFrame) -> float:
    scored = _trading(rows)
    predicted = np.clip(np.asarray(pipeline.predict(scored), dtype="float64"), 0.0, None)
    return metrics.r2(scored[s.SALES], predicted)
