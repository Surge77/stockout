"""The search spaces, as literals and nothing else.

Separated from the code that searches them so this file can be read as data: no imports
beyond typing, no logic, nothing that can fail at runtime. A reviewer wanting to know what
`ridge` was allowed to consider reads eight lines here rather than tracing a builder.

**Every key is a path through the pipeline** the adapter assembles, which is
`preprocess -> compact -> estimate`. So `estimate__alpha` is the estimator's own `alpha`,
and `estimate__regressor__C` reaches through a `TransformedTargetRegressor` to the model
inside it. Getting that prefix wrong is the classic silent failure of a grid search: the
parameter is rejected, sklearn raises, and the temptation is to delete the line rather than
fix the path.

**The grids are small on purpose.** Every candidate is multiplied by the number of
cross-validation folds and then by the training rows, and a search that cannot be rerun is
not reproducible evidence. Three to six values per parameter, chosen to span orders of
magnitude rather than to fill a range finely: the difference between `alpha=0.1` and
`alpha=1` is worth knowing, the difference between `0.1` and `0.15` is noise on this data.
"""

from __future__ import annotations

from typing import Final

#: Regularisation strengths, spanning four orders of magnitude. A grid that brackets the
#: chosen value on both sides is the only kind that answers "is this a corner solution".
_ALPHAS: Final[tuple[float, ...]] = (0.01, 0.1, 1.0, 10.0, 100.0)

#: The L1 models get a different, higher range, and the reason is worth stating because
#: it looks like an inconsistency.
#:
#: `alpha` is not scale-free: it weighs a penalty against a squared-error loss, and this
#: target is daily revenue in the thousands, so the loss term is enormous and an alpha of
#: 0.01 is indistinguishable from no penalty at all. Ridge tolerates that — it has a
#: closed form and stays well conditioned. Lasso and ElasticNet are solved by coordinate
#: descent, which on fifty correlated calendar columns with a near-zero penalty does not
#: converge: measured at a duality gap of 3.2e7 against a tolerance of 3.3e5, after
#: 20,000 iterations. `filterwarnings = ["error"]` turns that into a failed test, which
#: is the pyproject setting doing precisely its job.
#:
#: Raising the iteration cap further would spend minutes to arrive at an unregularised
#: fit that `linear` already provides. Searching a range where the penalty means
#: something is the better answer.
_L1_ALPHAS: Final[tuple[float, ...]] = (1.0, 10.0, 100.0, 1000.0)

REGRESSION_GRIDS: Final[dict[str, dict[str, list[object]]]] = {
    "ridge": {"estimate__alpha": list(_ALPHAS)},
    "lasso": {"estimate__alpha": list(_L1_ALPHAS)},
    "elastic_net": {
        "estimate__alpha": list(_L1_ALPHAS),
        # 0.15 is nearly Ridge, 0.85 is nearly Lasso. Sweeping the middle is what makes
        # ElasticNet a third model rather than a slower copy of one of the other two.
        "estimate__l1_ratio": [0.15, 0.5, 0.85],
    },
    "polynomial": {"estimate__fit__alpha": [1.0, 10.0, 100.0]},
    "decision_tree": {
        "estimate__max_depth": [4, 8, 12, 20],
        "estimate__min_samples_leaf": [10, 50, 200],
    },
    "random_forest": {
        "estimate__max_depth": [8, 12, 20],
        "estimate__min_samples_leaf": [20, 50],
    },
    "hist_gradient_boosting": {
        "estimate__max_depth": [4, 8, 12],
        "estimate__learning_rate": [0.05, 0.1, 0.2],
    },
    # Reaches through the TransformedTargetRegressor that puts the target on a log scale.
    "knn": {
        "estimate__regressor__n_neighbors": [5, 15, 30],
        "estimate__regressor__weights": ["uniform", "distance"],
    },
    "svr": {
        "estimate__regressor__C": [0.1, 1.0, 10.0],
        "estimate__regressor__epsilon": [0.01, 0.1],
    },
}

CLASSIFICATION_GRIDS: Final[dict[str, dict[str, list[object]]]] = {
    # C is the inverse of regularisation strength, so this brackets the same range as
    # _ALPHAS does for the regressors, read from the other end.
    "logistic": {"estimate__C": [0.01, 0.1, 1.0, 10.0]},
    "decision_tree": {
        "estimate__max_depth": [4, 8, 12, 20],
        "estimate__min_samples_leaf": [10, 50, 200],
    },
    "random_forest": {
        "estimate__max_depth": [8, 12, 20],
        "estimate__min_samples_leaf": [20, 50],
    },
    "hist_gradient_boosting": {
        "estimate__max_depth": [4, 8, 12],
        "estimate__learning_rate": [0.05, 0.1, 0.2],
    },
    "knn": {
        "estimate__n_neighbors": [5, 15, 30],
        "estimate__weights": ["uniform", "distance"],
    },
    "svc": {"estimate__C": [0.1, 1.0, 10.0], "estimate__gamma": ["scale", "auto"]},
    "linear_svc": {"estimate__C": [0.1, 1.0, 10.0]},
}

_BY_TASK: Final[dict[str, dict[str, dict[str, list[object]]]]] = {
    "regression": REGRESSION_GRIDS,
    "classification": CLASSIFICATION_GRIDS,
}


def grid_for(name: str, *, task: str) -> dict[str, list[object]]:
    """The search space for one model, or an empty one if it has nothing to tune.

    Empty rather than an error: `dummy` and `linear` have no hyperparameters worth
    searching, and `tune --model linear` should say "nothing to search" rather than
    behave as though the model were unknown.
    """
    return dict(_BY_TASK.get(task, {}).get(name, {}))


def candidate_count(grid: dict[str, list[object]]) -> int:
    """How many fits one pass over this grid costs, before folds are multiplied in.

    Printed before a search starts. A grid that turns out to be 240 candidates is worth
    knowing about while it can still be made smaller.
    """
    total = 1
    for values in grid.values():
        total *= len(values)
    return total if grid else 0
