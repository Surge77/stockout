"""Fit both models on one split and save them with the thresholds they need.

Both, not one: the product answers two questions per store-day, and they are separate
models. They share the same split so their scores are comparable, and the demand
thresholds are fitted on the training rows of that split alone.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import r2_score

from .config import MODEL_PATH, RANDOM_SEED, TEST_SIZE
from .data import schemas as s
from .datasets import Prepared, prepare
from .evaluate import score, scored_against, split, training_rows
from .features import ENTITY_CODE, PRODUCT_FEATURES
from .persistence import Artifact, save
from .registry import build

#: What `train` fits when nothing is named. Gradient boosting handles the promo/holiday
#: interactions that the linear models flatten; random forest is the strongest taught
#: classifier on this shape. Both are overridable per run.
DEFAULT_REGRESSOR = "gradient_boosting"
DEFAULT_CLASSIFIER = "random_forest"

#: How many stores the leave-one-store-out score averages over. Five is enough to be
#: stable to two decimals and cheap enough to run on every `train`.
_COLD_SAMPLE_STORES = 5

#: Features the cold-start model may not use, and why each one is out.
#:
#: `entity_code` is the obvious one: a store nobody has seen has no id. `store_code` and
#: `item_code` are the same thing wearing two hats — on a product source they are the two
#: halves of that id, and leaving them in would let the model answer a described item by
#: recognising which item it is.
#:
#: The three ages are subtler and were found by measuring. They are per-store constants
#: plus a time trend — `competition_open_age_months` alone carries 16% of the deployed
#: model's importance — so they act as a second identity channel. Worse, a described shop
#: has no value for them: nobody filling in a form knows when a hypothetical competitor
#: opened. They arrive as zero, which never occurs in training, and the model is then
#: evaluated somewhere it has never been. Two clearly different shops came back within
#: 4% of each other because both rows fell down the same degenerate branch.
#:
#: What is left is exactly what a person can actually state about a shop that does not
#: exist yet: the calendar, the day's promotions, and the store's own attributes.
_COLD_EXCLUDED: tuple[str, ...] = (
    ENTITY_CODE,
    *PRODUCT_FEATURES,
    "competition_open_age_months",
    "promo2_age_weeks",
    "is_promo2_month",
)


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
    sales_model.fit(*training_rows(regressor, "regression", regression, seed=seed))

    # Same cap as the comparison table. Deploying `--classifier svc` against Rossmann
    # would otherwise fit on 800,000 rows at quadratic cost and never return.
    demand_model = build(classifier, task="classification", seed=seed)
    demand_model.fit(*training_rows(classifier, "classification", classification, seed=seed))

    cold_model, cold_columns, cold_scores = _fit_cold_start(regression, regressor, seed)

    observed_from, observed_through = prepared.date_range()

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
        unit=prepared.source.unit,
        has_products=prepared.source.has_products,
        entities=prepared.entities(),
        entity_vocabulary=prepared.entity_vocabulary,
        # Scored through `scored_against`, which puts the store's line back before
        # measuring. The regressor answers in residuals on a de-trended split, and an r²
        # measured on residuals is a different, easier number wearing the same name.
        scores={
            "regression": score(
                "regression", *scored_against(sales_model, "regression", regression)
            ),
            "classification": score(
                "classification", *scored_against(demand_model, "classification", classification)
            ),
        },
        trends=regression.trends,
        n_train=len(regression.X_train),
        n_test=len(regression.X_test),
        cold_regressor=cold_model,
        cold_feature_columns=cold_columns,
        cold_scores=cold_scores,
        observed_from=observed_from,
        observed_through=observed_through,
    )
    return artifact, save(artifact, path)


def _fit_cold_start(regression, regressor: str, seed: int):
    """Fit the regressor again without `entity_code`, and score it on a store it never saw.

    Returns nothing at all for a source that describes its entities not at all. The model
    answers *how much would one like this take*, and "like this" has to be spelled out in
    something: store type, assortment and competitor distance for a shop, category for an
    item. `store_item_demand` publishes none of either — Kaggle never said what any item is
    — so the only features left would be the calendar, every description would arrive as
    the same row, and the page would return one confident number for every question anybody
    could ask it. An absent model the surface can explain beats a fitted one that cannot
    tell two questions apart.

    Measured on `synthetic_products`, holding out each of one shop's twenty items in turn:
    the calendar alone scores a median r² of -3.85, and adding the category -0.37. Still
    negative — a category is not an item — but a different order of wrong, and better on
    fifteen of the twenty.

    Two different questions hide behind "predict this store's sales". *How much will
    store 12 take on Monday* may use everything known about store 12, identity included.
    *How much would a shop like this take* may not — there is no id to supply, and a model
    trained with one has no honest value to put there.

    The reported score is deliberately not the ordinary held-out r². It is a leave-one-
    store-out score: fit without a store entirely, then predict it. That is the question
    this model exists to answer, and the only number that describes how well it does.
    """
    prepared = regression.prepared
    if prepared.stores is None:
        return None, [], {}

    columns = [c for c in prepared.feature_columns if c not in _COLD_EXCLUDED]

    model = build(regressor, task="regression", seed=seed)
    model.fit(regression.X_train[columns], regression.Y_train)

    return model, columns, {"unseen_store_r2": _unseen_store_r2(prepared, columns, regressor, seed)}


def _unseen_store_r2(prepared: Prepared, columns: list[str], regressor: str, seed: int) -> float:
    """Mean r² over a few stores, each held out of training entirely.

    A sample rather than every store: this runs on every `train`, and fitting once per
    store would turn a four-second command into a four-minute one for a number that moves
    in the third decimal.
    """
    frame = prepared.frame
    entities = frame[s.ENTITY].astype("string")
    sample = prepared.entities()[:: max(1, len(prepared.entities()) // _COLD_SAMPLE_STORES)]

    scores: list[float] = []
    for held in sample[:_COLD_SAMPLE_STORES]:
        train_rows = frame[entities != str(held)]
        test_rows = frame[entities == str(held)]
        if train_rows.empty or test_rows.empty:
            continue
        model = build(regressor, task="regression", seed=seed)
        model.fit(train_rows[columns], train_rows[s.SALES])
        scores.append(float(r2_score(test_rows[s.SALES], model.predict(test_rows[columns]))))

    return float(sum(scores) / len(scores)) if scores else 0.0
