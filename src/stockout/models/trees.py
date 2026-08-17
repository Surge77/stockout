"""Trees and the two ways of combining them, for both tasks.

The linear family assumes the response is a weighted sum. Retail demand is not: a
promotion on a Saturday in December is worth more than the three effects added up, and a
tree finds that interaction without being told to look for it. This is where the
regression half stops being a straight line and the classification half gets its
strongest models.

**DecisionTree** alone, so the ensembles have something to be compared against. Depth is
capped: an unconstrained tree on 800k rows memorises the training window, and the
depth-versus-accuracy curve in the notebook is the clearest picture of overfitting the
project has.

**Bagging** fits many trees on bootstrap resamples and averages them. **RandomForest** is
bagging plus one more idea — each split considers a random subset of the features — and
the pair is here precisely so that the difference between the two isolates what that one
extra idea contributes. Neither is a black box if you can say what separates them.

**HistGradientBoosting** replaces the LightGBM that used to live here (ADR 0018). Same
family, no extra dependency, native handling of the missing values that
`competition_distance` ships with, and fast on this many rows because it bins the
features first.

Every count here is capped below the library defaults. `n_estimators=100` on 800k rows is
tens of minutes and several gigabytes of tree nodes, and a comparison table nobody can
regenerate is not evidence. The caps are stated rather than tuned.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .spec import ModelSpec

#: Deep enough to find weekday-by-promotion-by-season interactions, shallow enough not to
#: carve out individual days. The notebook sweeps this; the value here is the sweep's answer.
_MAX_DEPTH = 12

#: Half the library default. Fifty trees is within a percent of a hundred on this data and
#: takes half the time and half the memory, which is the trade a laptop wants.
_N_ESTIMATORS = 50

#: No leaf may rest on fewer rows than this. The single most effective guard against a
#: forest memorising one store's one strange Tuesday.
_MIN_LEAF = 50

#: Trees parallelise cleanly across cores. The grid search deliberately does not — see
#: `models/tuning.py` — so the parallelism lives here where it costs nothing to pickle.
_N_JOBS = -1


def _decision_tree_regressor(seed: int) -> Any:
    return DecisionTreeRegressor(
        max_depth=_MAX_DEPTH, min_samples_leaf=_MIN_LEAF, random_state=seed
    )


def _random_forest_regressor(seed: int) -> Any:
    return RandomForestRegressor(
        n_estimators=_N_ESTIMATORS,
        max_depth=_MAX_DEPTH,
        min_samples_leaf=_MIN_LEAF,
        n_jobs=_N_JOBS,
        random_state=seed,
    )


def _bagging_regressor(seed: int) -> Any:
    """Bagging over the *same* base tree the forest uses, so the comparison is clean.

    Left to its default estimator this would bag unconstrained trees and lose to the
    forest for two reasons at once — no feature subsampling *and* no depth cap — and the
    notebook could not attribute the gap to either.
    """
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=_MAX_DEPTH, min_samples_leaf=_MIN_LEAF),
        n_estimators=_N_ESTIMATORS,
        n_jobs=_N_JOBS,
        random_state=seed,
    )


def _hist_gradient_boosting_regressor(seed: int) -> Any:
    # `early_stopping` is left at "auto", which turns it on above 10,000 rows. When it
    # is on it carves its own validation split out of the training rows, so the seed is
    # load-bearing rather than decoration: without it two runs of the same command
    # produce different scores.
    return HistGradientBoostingRegressor(max_depth=_MAX_DEPTH, random_state=seed)


def _decision_tree_classifier(seed: int) -> Any:
    return DecisionTreeClassifier(
        max_depth=_MAX_DEPTH,
        min_samples_leaf=_MIN_LEAF,
        class_weight="balanced",
        random_state=seed,
    )


def _random_forest_classifier(seed: int) -> Any:
    return RandomForestClassifier(
        n_estimators=_N_ESTIMATORS,
        max_depth=_MAX_DEPTH,
        min_samples_leaf=_MIN_LEAF,
        class_weight="balanced",
        n_jobs=_N_JOBS,
        random_state=seed,
    )


def _bagging_classifier(seed: int) -> Any:
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=_MAX_DEPTH, min_samples_leaf=_MIN_LEAF, class_weight="balanced"
        ),
        n_estimators=_N_ESTIMATORS,
        n_jobs=_N_JOBS,
        random_state=seed,
    )


def _hist_gradient_boosting_classifier(seed: int) -> Any:
    return HistGradientBoostingClassifier(max_depth=_MAX_DEPTH, random_state=seed)


TREE_REGRESSORS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="decision_tree",
        task="regression",
        build=_decision_tree_regressor,
        note="one tree; the thing the ensembles are an argument against",
    ),
    ModelSpec(
        name="bagging",
        task="regression",
        build=_bagging_regressor,
        note="many trees on resampled rows, averaged",
    ),
    ModelSpec(
        name="random_forest",
        task="regression",
        build=_random_forest_regressor,
        note="bagging plus a random feature subset per split — the gap isolates that idea",
    ),
    ModelSpec(
        name="hist_gradient_boosting",
        task="regression",
        build=_hist_gradient_boosting_regressor,
        note="boosted trees on binned features; handles missing values natively",
    ),
)

TREE_CLASSIFIERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="decision_tree",
        task="classification",
        build=_decision_tree_classifier,
        note="one tree, class-weighted against the test window's drift",
    ),
    ModelSpec(
        name="bagging",
        task="classification",
        build=_bagging_classifier,
        note="many trees on resampled rows, voted",
    ),
    ModelSpec(
        name="random_forest",
        task="classification",
        build=_random_forest_classifier,
        note="bagging plus a random feature subset per split",
    ),
    ModelSpec(
        name="hist_gradient_boosting",
        task="classification",
        build=_hist_gradient_boosting_classifier,
        note="boosted trees; usually the strongest classifier here",
    ),
)
