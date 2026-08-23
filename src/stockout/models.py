"""The registered estimators, each wrapped in the Pipeline shape the notebooks use.

Every model is `Pipeline([("Scaler", StandardScaler()), ("Model", estimator)])`. Scaling
is a no-op for the tree models and is applied to them anyway, because one pattern is
easier to read than two and the cost is nil.

The regression hyperparameters are the assignment's, not invented here:
`Ridge(alpha=1)` and `Lasso(alpha=100)` are the values on pages 4 and 3 of
`Task_Regression.pdf`, and `PolynomialFeatures(degree=2)` is page 2.

**Three regressors here are outside the course material and are marked as such in
the registry.** `RandomForestRegressor` is the regression form of a classifier the
notebooks do teach; `GradientBoostingRegressor` and `VotingRegressor` appear in no
notebook in any form. They were added by request after 0.7.0 removed the untaught
models, so the exception is recorded rather than quietly folded in — see the
`taught` field below and the note in README.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import Task

RIDGE_ALPHA = 1.0
LASSO_ALPHA = 100.0
POLYNOMIAL_DEGREE = 2

#: LogisticRegression's default of 100 is not enough for six unscaled-then-scaled
#: features here and it warns rather than failing. Raised until it converges quietly.
LOGISTIC_MAX_ITER = 1_000

#: The three estimators `voting` averages. Deliberately unalike — a vote between three
#: linear models would be a linear model with extra steps. Each is already registered
#: here on its own, so the table shows what the ensemble was built from.
VOTING_MEMBERS = ("linear", "random_forest", "gradient_boosting")


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the registry: a name, its task, and how to build it."""

    name: str
    task: Task
    build: Callable[[int], Pipeline]
    """Given a random seed, return an unfitted pipeline. Called once per run."""

    note: str = ""

    taught: bool = True
    """Whether this estimator appears in the course notebooks.

    False for the three regressors added after 0.7.0. Printed in the comparison table,
    because a reader comparing a taught model against an untaught one should be told
    which is which rather than having to know."""

    max_training_rows: int | None = None
    """Fit this estimator on at most this many rows, sampling if the split is larger.

    Only kernel `SVC` sets it, and only because its training cost is quadratic in the
    number of rows. It takes about fifty seconds on the synthetic split's 37,000; Rossmann
    trains on roughly 800,000, which is over four hundred times the work — hours, for one
    cell of one table.

    Dropping it instead was the obvious alternative and is the wrong one: `svc` is a
    *taught* model, and a comparison table that quietly omits the slow taught model while
    keeping the fast untaught ones is answering a different question than it claims to.

    The cost is that the capped row is no longer a like-for-like comparison, so
    `evaluate` reports `n_fit` for every model rather than hiding the difference in a
    footnote. A reader can then see that `svc` lost to `random_forest` on a twentieth of
    the data and judge what that is worth."""


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("Scaler", StandardScaler()), ("Model", estimator)])


def _linear(seed: int) -> Pipeline:
    return _pipeline(LinearRegression())


def _ridge(seed: int) -> Pipeline:
    return _pipeline(Ridge(alpha=RIDGE_ALPHA))


def _lasso(seed: int) -> Pipeline:
    return _pipeline(Lasso(alpha=LASSO_ALPHA))


def _polynomial(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("Poly", PolynomialFeatures(degree=POLYNOMIAL_DEGREE)),
            ("Scaler", StandardScaler()),
            ("Model", LinearRegression()),
        ]
    )


def _random_forest_regressor(seed: int) -> Pipeline:
    return _pipeline(RandomForestRegressor(random_state=seed))


def _gradient_boosting(seed: int) -> Pipeline:
    return _pipeline(GradientBoostingRegressor(random_state=seed))


def _voting(seed: int) -> Pipeline:
    return _pipeline(
        VotingRegressor(
            [
                ("linear", LinearRegression()),
                ("random_forest", RandomForestRegressor(random_state=seed)),
                ("gradient_boosting", GradientBoostingRegressor(random_state=seed)),
            ]
        )
    )


def _logistic(seed: int) -> Pipeline:
    return _pipeline(LogisticRegression(max_iter=LOGISTIC_MAX_ITER, random_state=seed))


def _knn(seed: int) -> Pipeline:
    return _pipeline(KNeighborsClassifier())


def _decision_tree(seed: int) -> Pipeline:
    return _pipeline(DecisionTreeClassifier(random_state=seed))


def _random_forest(seed: int) -> Pipeline:
    return _pipeline(RandomForestClassifier(random_state=seed))


def _bagging(seed: int) -> Pipeline:
    return _pipeline(BaggingClassifier(random_state=seed))


#: The cap on `svc`'s training rows. Chosen as the largest round number that keeps one
#: fit inside about two minutes on the machine this was developed on — a comparison table
#: nobody waits for is a table nobody reads. Comfortably above the synthetic split, so the
#: cap is inert there and the published synthetic figures are unaffected by its existence.
SVC_MAX_ROWS = 50_000


def _svc(seed: int) -> Pipeline:
    return _pipeline(SVC(random_state=seed))


#: Simplest first, and the comparison table keeps this order. A table sorted by score
#: answers "which won"; this order also answers "did the extra complexity pay".
REGRESSORS: tuple[ModelSpec, ...] = (
    ModelSpec("linear", "regression", _linear, "plain least squares"),
    ModelSpec("ridge", "regression", _ridge, f"L2, alpha={RIDGE_ALPHA:g}"),
    ModelSpec("lasso", "regression", _lasso, f"L1, alpha={LASSO_ALPHA:g}"),
    ModelSpec("polynomial", "regression", _polynomial, f"degree {POLYNOMIAL_DEGREE}"),
    ModelSpec(
        "random_forest", "regression", _random_forest_regressor, "100 trees", taught=False
    ),
    ModelSpec(
        "gradient_boosting", "regression", _gradient_boosting, "100 stages", taught=False
    ),
    ModelSpec(
        "voting",
        "regression",
        _voting,
        f"averages {', '.join(VOTING_MEMBERS)}",
        taught=False,
    ),
)

CLASSIFIERS: tuple[ModelSpec, ...] = (
    ModelSpec("logistic", "classification", _logistic, "linear decision boundary"),
    ModelSpec("knn", "classification", _knn, "distance to the 5 nearest rows"),
    ModelSpec("decision_tree", "classification", _decision_tree, "one tree"),
    ModelSpec("random_forest", "classification", _random_forest, "100 trees, bagged"),
    ModelSpec("bagging", "classification", _bagging, "10 trees on bootstrap samples"),
    ModelSpec("svc", "classification", _svc, "RBF kernel", max_training_rows=SVC_MAX_ROWS),
)
