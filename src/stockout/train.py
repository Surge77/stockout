"""Fit one model per task on everything available, and write the pair to disk.

`evaluate/comparison.py` answers "which model is best" by holding a window back. This
answers a different question — "what should be deployed" — and the difference is that
nothing is held back. Once the comparison has chosen, the chosen model should see every
row there is; the last six weeks of history are the most informative rows in the file and
withholding them from production to preserve a scoreboard would be superstition.

The scores stored alongside are therefore *not* from this fit. They come from the
held-out comparison, and they are carried into the artifact so that a served prediction
can still say what the model was measured at. A score computed on the rows a model was
fitted on is not a score.
"""

from __future__ import annotations

import pandas as pd

from .config import RANDOM_SEED
from .evaluate.comparison import compare
from .features.build import feature_columns
from .models.registry import build
from .persistence import Artifact, new_artifact
from .split.strategies import time_holdout
from .targets import add_demand_class, fit_thresholds

#: What gets deployed when the caller does not choose. Both won their half of the
#: comparison on the committed sample, and both are cheap enough to refit on demand.
DEFAULT_REGRESSOR = "hist_gradient_boosting"
DEFAULT_CLASSIFIER = "hist_gradient_boosting"


def train(
    frame: pd.DataFrame,
    *,
    horizon: int,
    regressor: str = DEFAULT_REGRESSOR,
    classifier: str = DEFAULT_CLASSIFIER,
    test_days: int = 28,
    gap_days: int | None = None,
    seed: int = RANDOM_SEED,
    score: bool = True,
) -> Artifact:
    """Score on a held-out window, then refit on everything and package the result."""
    gap = horizon if gap_days is None else gap_days
    scores = _held_out_scores(
        frame,
        regressor=regressor,
        classifier=classifier,
        test_days=test_days,
        gap_days=gap,
        seed=seed,
    ) if score else {}

    # Thresholds from the whole frame here, deliberately: this is the deployed labeller,
    # and it should describe every store-day known rather than an arbitrary earlier slice.
    thresholds = fit_thresholds(frame)
    labelled = add_demand_class(frame, thresholds)

    fitted_regressor = build(regressor, task="regression", horizon=horizon, seed=seed).fit(labelled)
    fitted_classifier = build(
        classifier, task="classification", horizon=horizon, seed=seed
    ).fit(labelled)

    return new_artifact(
        regressor=fitted_regressor,
        classifier=fitted_classifier,
        thresholds=thresholds,
        horizon=horizon,
        feature_columns=feature_columns(labelled),
        training_rows=fitted_regressor.training_rows,
        scores=scores,
    )


def _held_out_scores(
    frame: pd.DataFrame,
    *,
    regressor: str,
    classifier: str,
    test_days: int,
    gap_days: int,
    seed: int,
) -> dict[str, float]:
    """The numbers the artifact will quote, measured on rows the deployed fit then sees.

    That overlap is fine and is worth being explicit about: the score describes the
    *method* on unseen data, and the deployed model is the same method given more of it.
    What would not be fine is computing the score after refitting, which is what this
    exists to avoid doing by accident.
    """
    time_holdout(frame, test_days=test_days, gap_days=gap_days)

    regression = compare(
        frame, task="regression", test_days=test_days, gap_days=gap_days,
        models=[regressor], seed=seed,
    )
    classification = compare(
        frame, task="classification", test_days=test_days, gap_days=gap_days,
        models=[classifier], seed=seed,
    )
    return {
        "r2": float(regression["r2"].iloc[0]),
        "wmape": float(regression["wmape"].iloc[0]),
        "accuracy": float(classification["accuracy"].iloc[0]),
        "macro_f1": float(classification["macro_f1"].iloc[0]),
    }
