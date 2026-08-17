"""Logistic regression and the two baselines every classification table needs.

The tree and kernel classifiers live in `trees.py` and `kernels.py` beside their
regression twins, because what is worth knowing about a random forest is the same thing in
both tasks and splitting them by task would separate a docstring from half its subject.
What is left here is the linear classifier and the floors.

**LogisticRegression**, multinomial over three ordered classes. It is the classification
counterpart of `linear.py`'s floor: a weighted sum of features squashed into class
probabilities, and everything more complicated has to justify itself against it.

`class_weight="balanced"` matters more than it looks. On the *training* window the three
classes are a third each by construction, so weighting changes almost nothing there. On
the *test* window the distribution has drifted — measured at roughly 39/36/25 on the
synthetic sample — and the weighting is what stops the model quietly optimising for the
majority it happened to be trained on.

**DummyClassifier is not filler.** Three balanced classes put the floor at 33%, so an
accuracy of 71% is a real result and an accuracy of 38% is noise wearing a percentage
sign. Without the floor printed beside it, neither number can be read. `strategy="prior"`
predicts the commonest training class every time, which is the honest zero on this
problem — and on a drifted test window it will not even score 33%, which is itself worth
seeing.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression

from .spec import ModelSpec

#: lbfgs does not converge in the default 100 iterations on this many correlated calendar
#: columns, and `filterwarnings = ["error"]` turns that into a failed test rather than a
#: line of yellow text. Raised until the warning stopped.
_MAX_ITER = 5_000


def _logistic_regression(seed: int) -> Any:
    return LogisticRegression(
        max_iter=_MAX_ITER,
        class_weight="balanced",
        random_state=seed,
    )


def _dummy_classifier(seed: int) -> Any:
    return DummyClassifier(strategy="prior", random_state=seed)


def _dummy_regressor(_seed: int) -> Any:
    """Predicts the training mean, forever.

    The floor for R2 specifically: R2 is *defined* as the improvement over predicting the
    mean, so this model scores 0.0 on its own training distribution by construction. That
    it can score meaningfully *below* zero on a later window is the useful part — it means
    the mean itself has moved, which is a fact about the data rather than about any model.
    """
    return DummyRegressor(strategy="mean")


LINEAR_CLASSIFIERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="logistic",
        task="classification",
        build=_logistic_regression,
        drop_first=True,
        note="multinomial logistic regression; the linear floor for the classification half",
    ),
)

BASELINE_REGRESSORS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="dummy",
        task="regression",
        build=_dummy_regressor,
        note="predicts the training mean; R2 is defined as the improvement on this",
    ),
)

BASELINE_CLASSIFIERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="dummy",
        task="classification",
        build=_dummy_classifier,
        note="predicts the commonest training class; the 33% floor three classes imply",
    ),
)
