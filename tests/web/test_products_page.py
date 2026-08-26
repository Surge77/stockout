"""The shelf page, and the rule that decides which pages a deployment offers at all.

Every surface here reads what the deployed artefact *is* rather than what it is called. A
product deployment can rank a shelf and cannot describe a hypothetical item; a store-day
deployment is the reverse. Getting that from the dataset's name would work until somebody
added a fifth source.

The service is stubbed, as everywhere in `tests/web`: these assertions are about routing,
access control and what the HTML says, and fitting seven regressors per test would make
them take minutes to answer a question they are not asking. The arithmetic behind the
ranking is covered against a real artefact in `tests/test_products.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockout.ranking import RankedItem, Ranking
from stockout_web import config
from stockout_web.presentation import StoreProfile

GOOD_PASSWORD = "correct-horse-battery"

STORE = "3"
DATE = "2014-07-15"


def _ranked() -> Ranking:
    """A real `Ranking`, not a mock: the template reads five of its derived properties."""
    items = [
        RankedItem(
            entity=f"{STORE}_{item}",
            item=str(item),
            sales=sales,
            demand_class=demand,
            thresholds=(sales * 0.8, sales * 1.2),
            typical=sales * 0.9,
        )
        for item, sales, demand in ((7, 61.0, "High"), (2, 40.0, "Medium"), (5, 12.0, "Low"))
    ]
    return Ranking(store=STORE, date=pd.Timestamp(DATE), items=items, status=None)


class _StubCatalogue:
    def uses_promotions(self) -> bool:
        return False


class _StubBenchmarks:
    """The shelf page reads the measured floor to say how weak the band model is."""

    def class_floor(self) -> float:
        return 0.455


@dataclass
class _Artifact:
    dataset: str = "synthetic_products"
    entity_label: str = "store-item"
    regressor_name: str = "gradient_boosting"
    classifier_name: str = "random_forest"
    unit: str = "units"
    has_products: bool = True
    cold_regressor: object = None
    trained_at: str = "2026-01-01T00:00:00+00:00"
    n_train: int = 10
    n_test: int = 2
    entities: list[str] = field(default_factory=lambda: ["3_7", "3_2", "3_5"])
    thresholds: dict[str, tuple[float, float]] = field(default_factory=dict)
    cold_scores: dict[str, float] = field(default_factory=dict)
    cold_feature_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(
        default_factory=lambda: ["entity_code", "store_code", "item_code"]
    )
    scores: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "regression": {"r2": 0.92},
            "classification": {"accuracy": 0.57},
        }
    )

    def provenance(self) -> str:
        return "stub"


class _ProductService:
    """A deployment whose artefact has a product axis and no entity attributes."""

    artifact = _Artifact()
    catalogue = _StubCatalogue()
    benchmarks = _StubBenchmarks()

    def store_options(self) -> list[StoreProfile]:
        return [
            StoreProfile(
                entity="3_7", store_type="", assortment="", typical_sales=55.0,
                quiet_cut=48.0, busy_cut=66.0, competition_km=None, has_promo2=False,
                days=730, store=STORE, item="7", unit="units",
            )
        ]

    def item_options(self, store: str) -> list[StoreProfile]:
        return self.store_options()

    def stores(self) -> list[str]:
        return ["1", "2", "3"]

    def entity_for(self, store: str, item: str | None) -> str:
        return f"{store}_{item}"

    def entity_label(self) -> str:
        return "store-item"

    def unit(self) -> str:
        return "units"

    def has_products(self) -> bool:
        return True

    def supports_description(self) -> bool:
        return False

    def date_bounds(self) -> tuple[str, str]:
        return "2013-01-01", "2014-12-31"

    def rank_products(self, **_: object) -> Ranking:
        return _ranked()

    def cards(self) -> list[dict[str, object]]:
        return []


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The app on a throwaway database, deployed on a product artefact."""
    monkeypatch.setattr(config, "database_path", lambda: tmp_path / "users.db")
    monkeypatch.setenv("STOCKOUT_SESSION_SECRET", "test-secret-not-a-real-one")

    from stockout_web import main

    monkeypatch.setattr(main.ModelService, "load", classmethod(lambda cls: _ProductService()))
    with TestClient(main.app) as running:
        yield running


def _signed_in(client: TestClient) -> TestClient:
    client.post(
        "/register",
        data={"email": "shelf@example.com", "password": GOOD_PASSWORD,
              "confirm": GOOD_PASSWORD},
        follow_redirects=True,
    )
    return client


# -- access -------------------------------------------------------------------


def test_the_shelf_needs_an_account(client: TestClient) -> None:
    response = client.get("/products", follow_redirects=False)

    assert response.status_code in (302, 303, 401)


# -- the form -----------------------------------------------------------------


def test_the_form_lists_the_shops(client: TestClient) -> None:
    response = _signed_in(client).get("/products")

    assert response.status_code == 200
    assert "Shop 3" in response.text


def test_controls_the_models_never_saw_are_not_offered(client: TestClient) -> None:
    """A promotion checkbox on a dataset with no promotions invites input that does nothing."""
    body = _signed_in(client).get("/products").text

    assert 'name="promo"' not in body
    assert "records no promotions or holidays" in body


# -- the ranking --------------------------------------------------------------


def test_it_ranks_the_shelf_busiest_first(client: TestClient) -> None:
    response = _signed_in(client).post("/products", data={"store": STORE, "date": DATE})

    assert response.status_code == 200
    body = response.text
    assert body.index(">7<") < body.index(">2<") < body.index(">5<")


def test_it_shows_the_shelf_total(client: TestClient) -> None:
    body = _signed_in(client).post("/products", data={"store": STORE, "date": DATE}).text

    assert "113" in body  # 61 + 40 + 12


def test_each_row_carries_its_own_band(client: TestClient) -> None:
    """Sorted by units, judged per item — the two are allowed to disagree."""
    body = _signed_in(client).post("/products", data={"store": STORE, "date": DATE}).text

    for label in ("High", "Medium", "Low"):
        assert label in body


def test_the_figures_are_named_as_units(client: TestClient) -> None:
    body = _signed_in(client).post("/products", data={"store": STORE, "date": DATE}).text

    assert "units" in body


# -- what this deployment cannot answer ---------------------------------------


def test_the_described_shop_page_declines_and_says_why(client: TestClient) -> None:
    """No cold-start model exists, so the form must not be offered."""
    body = _signed_in(client).get("/what-if").text

    assert "publishes <strong>none of those</strong>" in body
    assert 'name="store_type"' not in body


def test_the_nav_offers_the_shelf_and_hides_the_described_shop(client: TestClient) -> None:
    body = _signed_in(client).get("/products").text

    assert 'href="/products"' in body
    assert 'href="/what-if"' not in body


def test_the_forecast_form_picks_a_shop_and_an_item(client: TestClient) -> None:
    """Two dropdowns, not one of every pair."""
    body = _signed_in(client).get("/").text

    assert 'name="store"' in body
    assert 'name="item"' in body


def test_the_band_model_is_reported_against_its_measured_floor(client: TestClient) -> None:
    """The floor is measured, not assumed.

    Per-entity terciles put it near a third *in training*; on a chronological hold-out the
    residual distribution drifts and it reached 45% on this data. A hardcoded 33% would
    have described a barely-above-chance classifier as a decent one.
    """
    body = _signed_in(client).post("/products", data={"store": STORE, "date": DATE}).text

    assert "for always guessing the commonest band" in body
    assert "57 times in 100" in body
