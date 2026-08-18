"""Fixtures for the web app: a throwaway database, a throwaway artefact, and a client.

Nothing here touches the real `artifacts/` directory. The user table and the model file
both live under `tmp_path`, so a test run cannot sign anybody out of a real instance or
overwrite a model somebody trained.

The artefact is built **once per session** and reused. Training fits two pipelines, and
doing it per test would put a minute on the suite for no extra coverage — every test that
needs a model needs the same one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stockout import config as ml_config
from stockout import persistence
from stockout.data import schemas as s
from stockout.dataset import prepare
from stockout.train import train
from stockout_web.main import create_app

#: Cheap and adequate. The point of these tests is the HTTP surface, not which estimator
#: won — `tests/test_comparison.py` covers that, and a boosted model would add seconds.
FAST_REGRESSOR = "ridge"
FAST_CLASSIFIER = "logistic"

ADMIN = ("admin@example.com", "admin-password")
MEMBER = ("member@example.com", "member-password")


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding one real trained artefact, built once for the whole session."""
    directory = tmp_path_factory.mktemp("artifacts")
    prepared = prepare()
    trained = train(
        prepared.frame,
        horizon=prepared.horizon,
        regressor=FAST_REGRESSOR,
        classifier=FAST_CLASSIFIER,
        test_days=28,
    )
    persistence.save(trained, directory / persistence.DEFAULT_ARTIFACT_NAME)
    return directory


@pytest.fixture
def app_env(
    tmp_path: Path, artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Point the app at a fresh database and the session's artefact directory."""
    database = tmp_path / "users.db"
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(database))
    # A fixed secret, so a cookie set in one request survives into the next within a test.
    monkeypatch.setenv("STOCKOUT_WEB_SECRET", "test-secret-not-used-anywhere-real")
    monkeypatch.delenv("STOCKOUT_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("STOCKOUT_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(ml_config, "ARTIFACT_DIR", artifact_dir)
    yield database


@pytest.fixture
def client(app_env: Path) -> Iterator[TestClient]:
    """A client whose lifespan has run, so the table exists and the model is loaded."""
    with TestClient(create_app(), follow_redirects=False) as test_client:
        yield test_client


@pytest.fixture
def unloaded_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A client with **no** artefact, which is the first-run state a fresh clone has.

    The app must still start and still let somebody register, or an admin can never sign
    in to train the model that is missing.
    """
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(tmp_path / "users.db"))
    monkeypatch.setenv("STOCKOUT_WEB_SECRET", "test-secret-not-used-anywhere-real")
    monkeypatch.delenv("STOCKOUT_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("STOCKOUT_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(ml_config, "ARTIFACT_DIR", tmp_path / "empty")
    with TestClient(create_app(), follow_redirects=False) as test_client:
        yield test_client


def register(client: TestClient, email: str, password: str) -> None:
    """Create an account through the public form and stay signed in as it."""
    response = client.post("/register", data={"email": email, "password": password})
    assert response.status_code == 303, response.text


def sign_in(client: TestClient, email: str, password: str) -> None:
    response = client.post("/login", data={"email": email, "password": password})
    assert response.status_code == 303, response.text


def sign_out(client: TestClient) -> None:
    assert client.post("/logout").status_code == 303


@pytest.fixture
def admin_client(client: TestClient) -> TestClient:
    """Signed in as an admin — the first account registered always is one."""
    register(client, *ADMIN)
    return client


@pytest.fixture
def member_client(client: TestClient) -> TestClient:
    """Signed in as a plain user, with an admin existing before them."""
    register(client, *ADMIN)
    sign_out(client)
    register(client, *MEMBER)
    return client


@pytest.fixture(scope="session")
def answerable_date() -> str:
    """A date the model can answer for: the day after history ends, inside the horizon."""
    return (prepare().frame[s.DATE].max() + pd.Timedelta(days=1)).date().isoformat()


@pytest.fixture(scope="session")
def unanswerable_date() -> str:
    """A date past `last_date + horizon`, which `predict` refuses rather than compounds."""
    return (prepare().frame[s.DATE].max() + pd.Timedelta(days=400)).date().isoformat()
