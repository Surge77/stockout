"""The linear family: the floor to beat, and the three ways of penalising it.

The course notebooks build these one at a time, each in its own file, each ending in an
R2. Here they are five entries in a registry that all share one preprocessor, because the
comparison is the point and a comparison is only fair if the only thing that changed is
the estimator.

**LinearRegression** is the floor. Every other number in the results table should be read
as "and what did the extra complexity buy over this".

**Ridge, Lasso and ElasticNet** are the same model with a penalty on the coefficients.
Ridge shrinks them, Lasso sets some to exactly zero, ElasticNet does both. `alpha` is not
guessed anywhere in this package — `models/grids.py` searches it, which is the difference
between a hyperparameter and a magic number. ADR 0017.

**Lasso's feature selection only means anything because `StandardScaler` ran first.** The
penalty is on coefficient magnitude, and an unscaled coefficient's magnitude is an
accident of the units its column happens to be in — metres of competitor distance against
a 0/1 promotion flag. That is the reason scaling is not optional here, and it is a better
answer than "the textbook says to".

`max_iter` is raised well above the default on the coordinate-descent models. They do not
converge in 1,000 iterations on this many correlated calendar columns, and `pyproject`
sets `filterwarnings = ["error"]`, so a ConvergenceWarning is a failed test rather than a
line of yellow text nobody reads.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .spec import ModelSpec

#: Coordinate descent needs room on this design matrix. Chosen by raising it until the
#: warning stopped rather than by taste.
_MAX_ITER = 20_000

#: Second-degree interactions only. `PolynomialFeatures(2)` on the full ~50-column matrix
#: would produce roughly 1,300 columns and on 800k rows that is a memory bomb for no
#: benefit — the one-hot columns are binary, so their squares are themselves and their
#: products are three-way indicators that a tree finds more cheaply. The expansion is
#: applied after the shared preprocessor and kept honest by the penalty that follows it.
_POLY_DEGREE = 2


def _linear(_seed: int) -> Any:
    return LinearRegression()


def _ridge(_seed: int) -> Any:
    return Ridge(alpha=1.0)


def _lasso(_seed: int) -> Any:
    # `selection="random"` converges faster on wide correlated matrices and takes a seed,
    # which is why this one is passed through rather than ignored.
    return Lasso(alpha=1.0, max_iter=_MAX_ITER, selection="random", random_state=_seed)


def _elastic_net(_seed: int) -> Any:
    return ElasticNet(
        alpha=1.0, l1_ratio=0.5, max_iter=_MAX_ITER, selection="random", random_state=_seed
    )


def _polynomial(_seed: int) -> Any:
    """Degree-2 expansion followed by a penalised fit.

    Ridge rather than a plain `LinearRegression`, because the expansion produces columns
    that are near-copies of one another — the square of a standardised feature correlates
    strongly with the feature — and an unpenalised fit on those is numerically unstable
    in a way that shows up as enormous coefficients of opposite sign.

    `interaction_only=False` keeps the squares: the point of the polynomial chapter is to
    show curvature being captured, and `days_since_start ** 2` is the one that does it.
    """
    return Pipeline(
        [
            ("expand", PolynomialFeatures(degree=_POLY_DEGREE, include_bias=False)),
            ("fit", Ridge(alpha=10.0)),
        ]
    )


LINEAR_REGRESSORS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="linear",
        task="regression",
        build=_linear,
        drop_first=True,
        note="the floor: ordinary least squares on every feature",
    ),
    ModelSpec(
        name="ridge",
        task="regression",
        build=_ridge,
        note="L2 penalty; shrinks correlated calendar columns toward each other",
    ),
    ModelSpec(
        name="lasso",
        task="regression",
        build=_lasso,
        note="L1 penalty; drives coefficients to exactly zero, so it also selects",
    ),
    ModelSpec(
        name="elastic_net",
        task="regression",
        build=_elastic_net,
        note="L1 and L2 together; selects like Lasso without discarding a whole group",
    ),
    ModelSpec(
        name="polynomial",
        task="regression",
        build=_polynomial,
        drop_first=True,
        note="degree-2 expansion then Ridge; the only model here that fits curvature",
    ),
)
