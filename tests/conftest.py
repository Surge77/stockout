"""Shared fixtures. The app on a throwaway database, with the models stubbed out.

Lives here rather than in one test module because two files now need it, and because the
stub is the reason the suite answers in seconds: these tests are about routing, access
control and rendering, not about what a model predicts, and fitting seven regressors per
test would make them take minutes to answer a question they are not asking.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stockout_web import config
from stockout_web.presentation import StoreProfile

GOOD_PASSWORD = "correct-horse-battery"


class _StubService:
    """Enough surface for the routes under test; none of them predicts anything."""

    @dataclass
    class _Artifact:
        dataset: str = "synthetic"
        entity_label: str = "store"
        regressor_name: str = "linear"
        classifier_name: str = "logistic"
        trained_at: str = "2026-01-01T00:00:00+00:00"
        n_train: int = 10
        n_test: int = 2
        entities: list[str] = field(default_factory=lambda: ["1"])
        thresholds: dict[str, tuple[float, float]] = field(default_factory=dict)
        #: What the described-shop page reports about the model it uses. Empty here —
        #: the page renders the absence as "not measured", which is a state it has to
        #: handle anyway.
        cold_scores: dict[str, float] = field(default_factory=dict)
        cold_feature_columns: list[str] = field(default_factory=list)
        scores: dict[str, dict[str, float]] = field(
            default_factory=lambda: {
                "regression": {"r2": 0.5},
                "classification": {"accuracy": 0.5},
            }
        )

        def provenance(self) -> str:
            return "stub"

    artifact = _Artifact()

    def entities(self) -> list[str]:
        return ["1"]

    def store_options(self) -> list[StoreProfile]:
        """The forecast form renders profiles, not bare ids, so the stub supplies one.

        A real `StoreProfile` rather than a mock: the template reads four of its derived
        properties, and a stub that answers every attribute would pass this test while the
        page was broken.
        """
        return [
            StoreProfile(
                entity="1",
                store_type="a",
                assortment="a",
                typical_sales=9_000.0,
                quiet_cut=8_000.0,
                busy_cut=10_000.0,
                competition_km=1.5,
                has_promo2=False,
                days=900,
            )
        ]

    def entity_label(self) -> str:
        return "store"

    def date_bounds(self) -> tuple[str, str]:
        return "2013-01-01", "2015-12-31"

    def cards(self) -> list[dict[str, object]]:
        return []

    async def comparison_tables(self) -> tuple[None, None, float]:
        """Startup warms this in the background; the stub answers instantly."""
        return None, None, 0.0


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An app on a throwaway database, with the models stubbed out."""
    database = tmp_path / "users.db"
    monkeypatch.setattr(config, "database_path", lambda: database)
    monkeypatch.setenv("STOCKOUT_SESSION_SECRET", "test-secret-not-a-real-one")

    from stockout_web import main

    monkeypatch.setattr(main.ModelService, "load", classmethod(lambda cls: _StubService()))
    with TestClient(main.app) as running:
        yield running


