"""Predicting a shop that does not exist, and the property that makes it possible.

The point of this module is not that the code runs. It is that a store's *attributes*
carry enough signal to answer without an id — which was false until `data/synth.py` was
changed to derive each store's level from its own metadata. These tests pin the property,
so that if the generator ever goes back to drawing levels at random the failure shows up
here rather than as a page quietly returning plausible nonsense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from stockout.data import schemas as s
from stockout.datasets import prepare
from stockout.hypothetical import StoreDescription, contributions, predict_described
from stockout.registry import build
from stockout.train import train

#: Below this, a described shop is not worth answering about and the page should not exist.
MIN_UNSEEN_STORE_R2 = 0.4


@pytest.fixture(scope="module")
def prepared():
    return prepare("synthetic")


@pytest.fixture(scope="module")
def artifact(prepared, tmp_path_factory):
    built, _ = train(prepared, path=tmp_path_factory.mktemp("artifacts") / "model.joblib")
    return built


def test_store_attributes_actually_move_sales(prepared) -> None:
    """The generator property everything else rests on.

    Assortment is the strongest of the three attribute effects. If a wider range stops
    being worth more than a basic one, the cold-start model has nothing to learn from.
    """
    level = prepared.frame.groupby(prepared.frame[s.ENTITY].astype(str))[s.SALES].mean()
    stores = prepared.stores.copy()
    stores[s.ENTITY] = stores[s.ENTITY].astype(str)
    joined = stores.set_index(s.ENTITY).join(level.rename("mean_sales"))

    by_assortment = joined.groupby(s.ASSORTMENT)["mean_sales"].mean()

    assert by_assortment["c"] > by_assortment["b"] > by_assortment["a"]


def test_a_nearer_competitor_means_lower_sales(prepared) -> None:
    stores = prepared.stores.copy()
    stores[s.ENTITY] = stores[s.ENTITY].astype(str)
    level = prepared.frame.groupby(prepared.frame[s.ENTITY].astype(str))[s.SALES].mean()
    joined = stores.set_index(s.ENTITY).join(level.rename("mean_sales"))

    distance = pd.to_numeric(joined[s.COMPETITION_DISTANCE], errors="coerce")
    known = distance.notna()
    correlation = float(np.corrcoef(distance[known], joined["mean_sales"][known])[0, 1])

    assert correlation > 0.2, "competitor distance carries no signal about a store's level"


def test_the_cold_model_never_sees_store_identity(artifact) -> None:
    """If `entity_code` leaks in, the model cannot answer about a new shop at all."""
    assert "entity_code" not in artifact.cold_feature_columns
    assert artifact.cold_regressor is not None


def test_the_cold_model_excludes_inputs_nobody_could_supply(artifact) -> None:
    """Ages of a competitor and of a promotion are unknowable for a shop that does not
    exist. Left in, they arrive as zero — a value absent from training — and two very
    different shops come back within a few percent of each other."""
    for column in ("competition_open_age_months", "promo2_age_weeks"):
        assert column not in artifact.cold_feature_columns


def test_it_can_predict_a_store_held_out_of_training(prepared, artifact) -> None:
    """The number the page reports, recomputed here on a store the fit never saw."""
    columns = artifact.cold_feature_columns
    entities = prepared.frame[s.ENTITY].astype(str)
    held = prepared.entities()[0]

    train_rows = prepared.frame[entities != str(held)]
    test_rows = prepared.frame[entities == str(held)]

    model = build("gradient_boosting", task="regression")
    model.fit(train_rows[columns], train_rows[s.SALES])
    scored = r2_score(test_rows[s.SALES], model.predict(test_rows[columns]))

    assert scored > MIN_UNSEEN_STORE_R2


def test_a_better_shop_is_predicted_to_sell_more(artifact) -> None:
    """A wide range far from competition should beat a basic range next door to one.

    This is the assertion a reader would make by eye, and it is the one that failed while
    the ages were still in the feature set: both shops came back within 4%.
    """
    strong = StoreDescription("a", "c", 20_000.0, promo2=True)
    weak = StoreDescription("d", "a", 200.0, promo2=False)

    better = predict_described(artifact, strong, date="2015-06-15", promo=True).sales
    worse = predict_described(artifact, weak, date="2015-06-15", promo=True).sales

    assert better > worse * 1.2


def test_contributions_name_every_described_input(artifact) -> None:
    described = StoreDescription("a", "c", 20_000.0, promo2=True)

    labels = {label for label, _ in contributions(artifact, described, date="2015-06-15")}

    assert labels == {"Assortment", "Store type", "Competitor distance", "Promo2", "Promotion"}


def test_a_promotion_never_lowers_the_estimate(artifact) -> None:
    described = StoreDescription("b", "b", 1_500.0, promo2=False)

    without = predict_described(artifact, described, date="2015-06-15", promo=False).sales
    with_promo = predict_described(artifact, described, date="2015-06-15", promo=True).sales

    assert with_promo > without
