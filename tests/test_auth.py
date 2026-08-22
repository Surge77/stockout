"""Auth and access control. The only tests in the package, and the reason is the blast radius.

Everywhere else a bug is a wrong number. Here a bug is a way in. These cover the six
things that would let somebody see or change what they should not, including the
first-user-becomes-admin race that a review caught in 0.6.0.

The model service is stubbed: these tests are about who may reach a route, not about what
the route predicts, and fitting seven regressors per test would make the file take minutes
to answer a question it is not asking.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stockout_web import config, db, users

REGISTER = "/register"
LOGIN = "/login"
ADMIN = "/admin"
ADMIN_USERS = "/admin/users"

GOOD_PASSWORD = "correct-horse-battery"
REJECTION_MESSAGE = "That email or password is wrong."
OTHER_PASSWORD = "another-good-password"


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

    def entity_label(self) -> str:
        return "store"

    def date_bounds(self) -> tuple[str, str]:
        return "2013-01-01", "2015-12-31"

    def cards(self) -> list[dict[str, object]]:
        return []


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


def _register(client: TestClient, email: str, password: str = GOOD_PASSWORD):
    return client.post(
        REGISTER, data={"email": email, "password": password}, follow_redirects=False
    )


def _login(client: TestClient, email: str, password: str = GOOD_PASSWORD):
    return client.post(
        LOGIN, data={"email": email, "password": password}, follow_redirects=False
    )


def test_first_account_becomes_admin_and_the_second_does_not(client: TestClient) -> None:
    _register(client, "first@example.com")
    _register(client, "second@example.com")

    with db.session() as connection:
        first = users.authenticate(connection, email="first@example.com", password=GOOD_PASSWORD)
        second = users.authenticate(
            connection, email="second@example.com", password=GOOD_PASSWORD
        )

    assert first is not None and first.role == "admin"
    assert second is not None and second.role == "user"


def test_rejects_a_wrong_password(client: TestClient) -> None:
    _register(client, "someone@example.com")
    client.post("/logout", follow_redirects=False)

    response = _login(client, "someone@example.com", "definitely-not-the-password")

    # 401 with the form re-rendered, not a redirect: an error in a query string
    # survives being shared and bookmarked.
    assert response.status_code == 401
    assert client.get(ADMIN, follow_redirects=False).status_code == 303


def test_an_unknown_email_gives_the_same_answer_as_a_wrong_password(
    client: TestClient,
) -> None:
    _register(client, "someone@example.com")
    client.post("/logout", follow_redirects=False)

    wrong_password = _login(client, "someone@example.com", "not-the-password")
    unknown_email = _login(client, "nobody@example.com", "not-the-password")

    # Same status and the same message. The pages differ only where the form echoes
    # back the address that was typed, which tells an attacker nothing they did not
    # already supply.
    assert wrong_password.status_code == unknown_email.status_code == 401
    assert REJECTION_MESSAGE in wrong_password.text
    assert REJECTION_MESSAGE in unknown_email.text


def test_a_normal_user_cannot_reach_the_admin_pages(client: TestClient) -> None:
    _register(client, "admin@example.com")
    client.post("/logout", follow_redirects=False)
    _register(client, "plain@example.com", OTHER_PASSWORD)

    for route in (ADMIN, ADMIN_USERS):
        assert client.get(route, follow_redirects=False).status_code == 403


def test_a_signed_out_visitor_is_sent_to_the_sign_in_page(client: TestClient) -> None:
    for route in ("/", ADMIN):
        response = client.get(route, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == LOGIN


def test_deactivating_an_account_takes_effect_on_the_next_request(
    client: TestClient,
) -> None:
    """The role and the active flag are re-read from the database every request, so a
    session that is already open does not survive being switched off."""
    _register(client, "admin@example.com")
    client.post("/logout", follow_redirects=False)
    _register(client, "plain@example.com", OTHER_PASSWORD)

    assert client.get("/", follow_redirects=False).status_code == 200

    with db.session() as connection:
        victim = users.authenticate(
            connection, email="plain@example.com", password=OTHER_PASSWORD
        )
        assert victim is not None
        users.set_active(connection, user_id=victim.id, is_active=False)

    assert client.get("/", follow_redirects=False).status_code == 303


def test_the_last_active_admin_cannot_be_demoted_or_deactivated(client: TestClient) -> None:
    _register(client, "only-admin@example.com")

    with db.session() as connection:
        admin = users.authenticate(
            connection, email="only-admin@example.com", password=GOOD_PASSWORD
        )
        assert admin is not None

        with pytest.raises(users.UserError):
            users.set_role(connection, user_id=admin.id, role="user")
        with pytest.raises(users.UserError):
            users.set_active(connection, user_id=admin.id, is_active=False)


def test_the_session_cookie_is_signed_and_http_only(client: TestClient) -> None:
    response = _register(client, "someone@example.com")

    cookie = response.headers.get("set-cookie", "")
    assert "httponly" in cookie.lower()
    # Starlette signs the payload, so the raw value is never readable json.
    assert "email" not in cookie


def test_a_short_password_is_refused(client: TestClient) -> None:
    response = _register(client, "someone@example.com", "short")

    assert response.status_code == 400
    with db.session() as connection:
        assert users.count(connection) == 0


def test_the_same_email_cannot_register_twice(client: TestClient) -> None:
    _register(client, "someone@example.com")
    client.post("/logout", follow_redirects=False)
    _register(client, "SOMEONE@example.com", OTHER_PASSWORD)

    with db.session() as connection:
        assert users.count(connection) == 1
