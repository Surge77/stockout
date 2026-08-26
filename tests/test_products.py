"""The product shape: a store-item entity, its two axes, and what the source cannot answer.

A product source asks a different question from a store-day one — *how many of item 25
will store 3 shift on Tuesday* rather than *what will store 3 take* — and it arrives with
less, not more: unit counts, no promotions, no holidays, and nothing at all describing what
any item is.

These tests pin the three places that difference has to be handled correctly rather than
averaged over. The entity carries two ids and both have to survive into the matrix. The
canonical columns the source does not have must not be offered to a model as features. And
the described-shop model has to be absent rather than fitted on nothing, with a page that
can say why.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout import features
from stockout.data import schemas as s
from stockout.data.adapters import source_for
from stockout.datasets import prepare
from stockout.errors import StockoutError
from stockout.hypothetical import StoreDescription, predict_described
from stockout.predict import predict, serving_row
from stockout.train import train

PRODUCTS = "synthetic_products"
STORE_DAY = "synthetic"
DATE = "2014-07-15"


@pytest.fixture(scope="module")
def prepared():
    return prepare(PRODUCTS)


@pytest.fixture(scope="module")
def artifact(prepared, tmp_path_factory):
    built, _ = train(prepared, path=tmp_path_factory.mktemp("artifacts") / "model.joblib")
    return built


# -- the adapter ---------------------------------------------------------------


def test_the_adapter_produces_every_canonical_column() -> None:
    """One shape for every source is what lets the registry and the panel be written once."""
    sales, stores = source_for(PRODUCTS).load()

    assert not set(s.CANONICAL_COLUMNS) - set(sales.columns)
    assert stores is None


def test_the_adapter_keeps_both_ids_beside_the_entity() -> None:
    """`entity` alone would make the two axes unrecoverable at training time."""
    sales, _ = source_for(PRODUCTS).load()
    row = sales.iloc[0]

    assert row[s.ENTITY] == f"{row[s.STORE]}{s.PRODUCT_ENTITY_SEPARATOR}{row[s.ITEM]}"


def test_the_source_declares_what_it_does_not_carry() -> None:
    """The surfaces read these rather than guessing from the dataset's name."""
    source = source_for(PRODUCTS)

    assert source.has_products
    assert not source.has_promotions
    assert not source.has_store_metadata
    assert source.unit == "units"


# -- the two axes --------------------------------------------------------------


def test_a_product_matrix_carries_both_axes(prepared) -> None:
    assert set(features.PRODUCT_FEATURES) <= set(prepared.feature_columns)


def test_a_store_day_matrix_does_not() -> None:
    """Rossmann's matrix must be untouched by the product work, or its scores move."""
    assert not set(features.PRODUCT_FEATURES) & set(prepare(STORE_DAY).feature_columns)


def test_the_axes_match_the_entity_they_came_from(prepared) -> None:
    frame = prepared.frame
    row = frame[frame[s.ENTITY] == "3_7"].iloc[0]

    assert (int(row["store_code"]), int(row["item_code"])) == (3, 7)


def test_a_serving_row_recovers_the_axes_from_the_entity_alone(artifact) -> None:
    """The request carries an entity and no `store`/`item` columns, so they are split out.

    Training reads the adapter's columns and serving splits the id; if the two disagreed,
    the model would be fitted on one thing and queried about another.
    """
    matrix = serving_row(artifact, entity="3_7", date=DATE)

    assert (int(matrix["store_code"].iloc[0]), int(matrix["item_code"].iloc[0])) == (3, 7)


def test_columns_the_source_lacks_are_not_offered_as_features(prepared) -> None:
    """A constant column is not a feature. The adapter supplies these to keep one shape."""
    offered = set(prepared.feature_columns)

    assert not offered & set(features.ROW_FEATURES)
    assert not offered & set(features.STORE_FEATURES)


def test_no_offered_feature_is_constant(prepared) -> None:
    constant = [c for c in prepared.feature_columns if features.is_constant(prepared.frame[c])]

    assert constant == []


# -- serving -------------------------------------------------------------------


def test_two_different_products_get_two_different_answers(artifact) -> None:
    """The regression guard for the entity-code bug, at the surface a visitor touches.

    Before the vocabulary travelled, every store-item request factorized to the same code
    and the model answered about whichever series sorted first — identically, for every
    product anyone asked about.
    """
    quiet = predict(artifact, entity="3_7", date=DATE)
    other = predict(artifact, entity="10_20", date=DATE)

    assert quiet.sales != other.sales


def test_an_unknown_product_is_refused_rather_than_answered(artifact) -> None:
    with pytest.raises(StockoutError, match="unknown store-item"):
        predict(artifact, entity="99_99", date=DATE)


def test_the_prediction_carries_the_products_own_cut_points(artifact) -> None:
    """Terciles are per series here too: busy *for this item in this shop*."""
    answer = predict(artifact, entity="3_7", date=DATE)
    low, high = answer.thresholds

    assert low < high
    assert answer.demand_class in ("Low", "Medium", "High")


# -- what the source cannot answer ---------------------------------------------


def test_no_cold_start_model_is_fitted_without_attributes(artifact) -> None:
    """Fitting one on the calendar alone would answer every described item identically."""
    assert artifact.cold_regressor is None
    assert artifact.cold_scores == {}


def test_describing_a_product_explains_itself_rather_than_guessing(artifact) -> None:
    """The message must not tell a visitor to retrain: no retrain will ever produce it."""
    with pytest.raises(StockoutError, match="describes no attributes"):
        predict_described(
            artifact,
            StoreDescription(
                store_type="a", assortment="a", competition_distance=500.0, promo2=False
            ),
            date=DATE,
        )


def test_the_artefact_records_what_its_numbers_count(artifact) -> None:
    """Serving holds an artefact and never the `Source`, so the wording has to travel."""
    assert artifact.unit == "units"
    assert artifact.has_products
    assert artifact.entity_label == "store-item"


def test_the_entity_vocabulary_travels(artifact, prepared) -> None:
    assert artifact.entity_vocabulary == prepared.entities()


def test_a_store_day_artefact_carries_no_vocabulary() -> None:
    """Numeric ids are their own code, so Rossmann's stores are never renumbered."""
    assert prepare(STORE_DAY).entity_vocabulary is None


def test_dates_past_the_data_are_still_answered(artifact) -> None:
    """Every series has its own line, so the level can go somewhere a tree cannot."""
    inside = predict(artifact, entity="3_7", date="2014-07-15")
    beyond = predict(artifact, entity="3_7", date="2015-07-15")

    assert pd.notna(beyond.sales)
    assert beyond.sales != inside.sales
