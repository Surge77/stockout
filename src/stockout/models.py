"""The ten taught estimators, each wrapped in the Pipeline shape the notebooks use.

Every model is `Pipeline([("Scaler", StandardScaler()), ("Model", estimator)])`. Scaling
is a no-op for the tree models and is applied to them anyway, because one pattern is
easier to read than two and the cost is nil.

The regression hyperparameters are the assignment's, not invented here:
`Ridge(alpha=1)` and `Lasso(alpha=100)` are the values on pages 4 and 3 of
`Task_Regression.pdf`, and `PolynomialFeatures(degree=2)` is page 2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
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


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the registry: a name, its task, and how to build it."""

    name: str
    task: Task
    build: Callable[[int], Pipeline]
    """Given a random seed, return an unfitted pipeline. Called once per run."""

    note: str = ""


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


def _svc(seed: int) -> Pipeline:
    return _pipeline(SVC(random_state=seed))


#: Simplest first, and the comparison table keeps this order. A table sorted by score
#: answers "which won"; this order also answers "did the extra complexity pay".
REGRESSORS: tuple[ModelSpec, ...] = (
    ModelSpec("linear", "regression", _linear, "plain least squares"),
    ModelSpec("ridge", "regression", _ridge, f"L2, alpha={RIDGE_ALPHA:g}"),
    ModelSpec("lasso", "regression", _lasso, f"L1, alpha={LASSO_ALPHA:g}"),
    ModelSpec("polynomial", "regression", _polynomial, f"degree {POLYNOMIAL_DEGREE}"),
)

CLASSIFIERS: tuple[ModelSpec, ...] = (
    ModelSpec("logistic", "classification", _logistic, "linear decision boundary"),
    ModelSpec("knn", "classification", _knn, "distance to the 5 nearest rows"),
    ModelSpec("decision_tree", "classification", _decision_tree, "one tree"),
    ModelSpec("random_forest", "classification", _random_forest, "100 trees, bagged"),
    ModelSpec("bagging", "classification", _bagging, "10 trees on bootstrap samples"),
    ModelSpec("svc", "classification", _svc, "RBF kernel"),
)
