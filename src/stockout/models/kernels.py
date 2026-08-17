"""KNN and the support-vector models: the two that cannot see the whole dataset.

Everything else in the registry trains on all ~800k rows. These do not, and pretending
otherwise is how a laptop ends up wedged the night before a deadline.

**The cost.** `SVR` and `SVC` with an RBF kernel are between quadratic and cubic in the
training rows, because they build a kernel matrix over pairs. 5,000 rows fits in seconds;
20,000 takes minutes; 50,000 takes hours and cannot survive a grid search wrapped around
it. `KNeighbors*` trains instantly and pays at predict time instead, so it tolerates ten
times as many rows before it hurts.

They are capped separately for that reason. Treating them as one "slow group" and giving
both the same budget would either starve KNN or hang the SVM.

**The cap is reported, not hidden.** `spec.sample_rows` travels with the model into the
comparison table, so a reader can see that the 0.71 next to `svr` was earned on 5,000
rows and the 0.88 next to `ridge` on all of them. A subsample that is declared is a
methodological decision; one that is silent is a fiddle. ADR 0015.

**The target is logged, and this is not optional.** `SVR`'s epsilon-insensitive tube has
a default half-width of 0.1 — meaning the model is indifferent to errors under 0.1 — and
`sales` runs to five figures. Left alone the tube is invisibly narrow, every point becomes
a support vector, the fit takes far longer and the output is close to garbage. Wrapping in
`TransformedTargetRegressor(log1p / expm1)` puts the target on a scale where 0.1 means
something, and has the side benefit of matching the multiplicative way retail demand
actually varies. `log1p` rather than `log` because a closed store's zero is a legitimate
value that `log` would send to negative infinity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from ..config import SUBSAMPLE_ROWS
from .spec import ModelSpec

#: Kernel models. The binding constraint in the whole registry.
_SVM_ROWS = SUBSAMPLE_ROWS

#: KNN trains in no time and pays at predict; ten times the SVM budget still runs in
#: about half a minute on a dense fifty-column matrix.
_KNN_ROWS = 50_000

#: Odd, so a three-class vote cannot tie two ways at once. Swept in the notebook.
_N_NEIGHBOURS = 15

#: Cache for the kernel matrix, in MB. The default 200 is small enough that sklearn
#: recomputes rows it has already seen; 512 is free on any machine that can run this.
_KERNEL_CACHE_MB = 512


def _log_target(estimator: Any) -> TransformedTargetRegressor:
    """Fit on `log1p(sales)` and invert on the way out. See the module docstring."""
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1
    )


def _knn_regressor(_seed: int) -> Any:
    # Distance weighting, because the neighbours of a given store-day are not equally
    # relevant: on a standardised matrix the nearest few are the same weekday under the
    # same promotion, and the fifteenth is a different season.
    return _log_target(
        KNeighborsRegressor(n_neighbors=_N_NEIGHBOURS, weights="distance", n_jobs=-1)
    )


def _svr(_seed: int) -> Any:
    return _log_target(SVR(kernel="rbf", C=1.0, epsilon=0.1, cache_size=_KERNEL_CACHE_MB))


def _knn_classifier(_seed: int) -> Any:
    return KNeighborsClassifier(n_neighbors=_N_NEIGHBOURS, weights="distance", n_jobs=-1)


def _svc(seed: int) -> Any:
    # `probability=True` is deliberately off. It would fit a second, internally
    # cross-validated model to calibrate the scores, multiplying an already expensive fit
    # by five for a probability nothing here consumes.
    return SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        cache_size=_KERNEL_CACHE_MB,
        random_state=seed,
    )


def _linear_svc(seed: int) -> Any:
    """The scalable alternative, kept so the cost of the kernel is visible.

    `SVC(kernel="linear")` and this solve the same problem by different routes, and the
    difference is the whole lesson: one is quadratic in the rows and one is linear, so
    this one sees every row the trees do. Whether the RBF kernel earns its subsample is
    then a question the results table answers rather than one the reader has to take on
    trust.
    """
    return SVC(kernel="linear", C=1.0, class_weight="balanced", random_state=seed)


KERNEL_REGRESSORS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="knn",
        task="regression",
        build=_knn_regressor,
        sample_rows=_KNN_ROWS,
        note="distance-weighted neighbours on a standardised matrix; log-transformed target",
    ),
    ModelSpec(
        name="svr",
        task="regression",
        build=_svr,
        sample_rows=_SVM_ROWS,
        note="RBF kernel on a 5,000-row sample; the tube is meaningless without log1p",
    ),
)

KERNEL_CLASSIFIERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="knn",
        task="classification",
        build=_knn_classifier,
        sample_rows=_KNN_ROWS,
        note="distance-weighted vote over 15 neighbours",
    ),
    ModelSpec(
        name="svc",
        task="classification",
        build=_svc,
        sample_rows=_SVM_ROWS,
        note="RBF kernel on a 5,000-row sample, class-weighted",
    ),
    ModelSpec(
        name="linear_svc",
        task="classification",
        build=_linear_svc,
        sample_rows=_KNN_ROWS,
        note="the same idea without the kernel trick, so the kernel's cost is visible",
    ),
)
