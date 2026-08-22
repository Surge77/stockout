"""Fit both models on one split and save them with the thresholds they need.

Both, not one: the product answers two questions per store-day, and they are separate
models. They share the same split so their scores are comparable, and the demand
thresholds are fitted on the training rows of that split alone.
"""

from __future__ import annotations

from pathlib import Path

from .config import MODEL_PATH, RANDOM_SEED, TEST_SIZE
from .datasets import Prepared, prepare
from .evaluate import score, split
from .persistence import Artifact, save
from .registry import build

#: What `train` fits when nothing is named. Gradient boosting handles the promo/holiday
#: interactions that the linear models flatten; random forest is the strongest taught
#: classifier on this shape. Both are overridable per run.
DEFAULT_REGRESSOR = "gradient_boosting"
DEFAULT_CLASSIFIER = "random_forest"


def train(
    dataset: str | Prepared,
    *,
    regressor: str = DEFAULT_REGRESSOR,
    classifier: str = DEFAULT_CLASSIFIER,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    path: Path = MODEL_PATH,
) -> tuple[Artifact, Path]:
    """Fit a regressor and a classifier, score both, and write one artefact."""
    prepared = prepare(dataset) if isinstance(dataset, str) else dataset

    regression = split(prepared, "regression", test_size=test_size, seed=seed)
    classification = split(prepared, "classification", test_size=test_size, seed=seed)

    sales_model = build(regressor, task="regression", seed=seed)
    sales_model.fit(regression.X_train, regression.Y_train)

    demand_model = build(classifier, task="classification", seed=seed)
    demand_model.fit(classification.X_train, classification.Y_train)

    artifact = Artifact(
        regressor=sales_model,
        classifier=demand_model,
        regressor_name=regressor,
        classifier_name=classifier,
        # Both splits use the same seed and the same rows, so the thresholds are
        # identical; taking them from the classification split names where they matter.
        thresholds=classification.thresholds,
        encoders=prepared.encoders,
        feature_columns=list(prepared.feature_columns),
        stores=prepared.stores,
        dataset=prepared.name,
        entity_label=prepared.source.entity_label,
        entities=prepared.entities(),
        scores={
            "regression": score(
                "regression", regression.Y_test, sales_model.predict(regression.X_test)
            ),
            "classification": score(
                "classification",
                classification.Y_test,
                demand_model.predict(classification.X_test),
            ),
        },
        n_train=len(regression.X_train),
        n_test=len(regression.X_test),
    )
    return artifact, save(artifact, path)
