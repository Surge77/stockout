"""The user module at its own level, and the seeding that happens before anyone logs in.

`test_auth.py` drives these through HTTP. This drives them directly, because the branches
that matter most are the refusals, and several of them cannot be reached through a form —
the role check, the 72-byte bcrypt limit, and the last-admin guard called with an id that
does not exist.

Project rule: auth logic carries 100% coverage. These are the cases the HTTP tests leave.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stockout import config as ml_config
from stockout_web import db, users
from stockout_web.main import create_app


@pytest.fixture
def connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A table of its own, with no HTTP anywhere near it."""
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(tmp_path / "users.db"))
    db.initialise()
    with db.session() as open_connection:
        yield open_connection


# --- passwords ---------------------------------------------------------------------


def test_a_password_over_bcrypts_limit_is_refused_rather_than_truncated() -> None:
    """bcrypt silently ignores everything past 72 bytes, so the last characters do nothing."""
    with pytest.raises(users.UserError, match="72 bytes"):
        users.hash_password("a" * 73)


def test_a_short_password_is_refused() -> None:
    with pytest.raises(users.UserError, match="8 characters"):
        users.hash_password("short")


def test_two_identical_passwords_get_different_hashes() -> None:
    """A per-password salt: identical passwords must not produce identical rows."""
    assert users.hash_password("the-same-password") != users.hash_password("the-same-password")


# --- emails ------------------------------------------------------------------------


@pytest.mark.parametrize("address", ["", "   ", "not-an-email", "@", "a b"])
def test_something_that_is_not_an_email_is_refused(address: str) -> None:
    with pytest.raises(users.UserError, match="email address"):
        users.normalise_email(address)


def test_an_email_is_trimmed_and_lowercased() -> None:
    assert users.normalise_email("  Person@Example.COM ") == "person@example.com"


# --- roles -------------------------------------------------------------------------


def test_creating_an_account_with_an_invented_role_is_refused(connection) -> None:  # type: ignore[no-untyped-def]
    """A role arriving from a form is data, and data does not get to invent permissions."""
    with pytest.raises(users.UserError, match="unknown role"):
        users.create(connection, email="x@example.com", password="a-password", role="superuser")  # type: ignore[arg-type]


def test_setting_an_invented_role_is_refused(connection) -> None:  # type: ignore[no-untyped-def]
    user = users.create(connection, email="x@example.com", password="a-password")
    with pytest.raises(users.UserError, match="unknown role"):
        users.set_role(connection, user_id=user.id, role="root")  # type: ignore[arg-type]


def test_the_schema_refuses_an_invented_role_too(connection) -> None:  # type: ignore[no-untyped-def]
    """Enforced twice: the CHECK constraint is the backstop for a bypassed helper."""
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO users (email, password_hash, role, is_active, created_at)"
            " VALUES (?, ?, ?, 1, ?)",
            ("y@example.com", "irrelevant", "root", "2026-01-01T00:00:00+00:00"),
        )


# --- missing accounts ---------------------------------------------------------------


def test_changing_the_role_of_a_missing_account_is_refused(connection) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(users.UserError, match="no such account"):
        users.set_role(connection, user_id=4321, role="admin")


def test_deactivating_a_missing_account_is_refused(connection) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(users.UserError, match="no such account"):
        users.set_active(connection, user_id=4321, is_active=False)


def test_by_id_returns_none_rather_than_raising(connection) -> None:  # type: ignore[no-untyped-def]
    assert users.by_id(connection, 4321) is None


# --- authentication ------------------------------------------------------------------


def test_authenticating_an_unknown_email_returns_none(connection) -> None:  # type: ignore[no-untyped-def]
    assert users.authenticate(connection, email="nobody@example.com", password="whatever") is None


def test_a_very_long_password_does_not_crash_authentication(connection) -> None:  # type: ignore[no-untyped-def]
    """bcrypt raises above 72 bytes; the supplied value is clipped before it gets there."""
    users.create(connection, email="x@example.com", password="a-real-password")
    assert users.authenticate(connection, email="x@example.com", password="a" * 500) is None


# --- the last admin -------------------------------------------------------------------


def test_the_last_admin_cannot_be_demoted(connection) -> None:  # type: ignore[no-untyped-def]
    admin = users.create(connection, email="a@example.com", password="a-password", role="admin")
    with pytest.raises(users.UserError, match="last active admin"):
        users.set_role(connection, user_id=admin.id, role="user")


def test_a_deactivated_admin_does_not_count_as_cover(connection) -> None:  # type: ignore[no-untyped-def]
    """Two admins, one deactivated, is one admin — and the guard has to know that."""
    first = users.create(connection, email="a@example.com", password="a-password", role="admin")
    second = users.create(connection, email="b@example.com", password="b-password", role="admin")
    users.set_active(connection, user_id=second.id, is_active=False)

    with pytest.raises(users.UserError, match="last active admin"):
        users.set_role(connection, user_id=first.id, role="user")


# --- seeding ---------------------------------------------------------------------------


def test_the_configured_admin_is_seeded_into_an_empty_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(tmp_path / "users.db"))
    monkeypatch.setenv("STOCKOUT_WEB_SECRET", "test-secret")
    monkeypatch.setenv("STOCKOUT_ADMIN_EMAIL", "seeded@example.com")
    monkeypatch.setenv("STOCKOUT_ADMIN_PASSWORD", "seeded-password")
    monkeypatch.setattr(ml_config, "ARTIFACT_DIR", tmp_path / "empty")

    with TestClient(create_app(), follow_redirects=False) as client:
        response = client.post(
            "/login", data={"email": "seeded@example.com", "password": "seeded-password"}
        )
        assert response.status_code == 303
        assert client.get("/admin").status_code == 200


def test_seeding_never_overwrites_an_existing_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise anybody who can set an env var resets the admin on a running system."""
    database = tmp_path / "users.db"
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(database))
    monkeypatch.setenv("STOCKOUT_WEB_SECRET", "test-secret")
    monkeypatch.setattr(ml_config, "ARTIFACT_DIR", tmp_path / "empty")

    db.initialise()
    with db.session() as connection:
        users.create(connection, email="first@example.com", password="first-password", role="admin")

    monkeypatch.setenv("STOCKOUT_ADMIN_EMAIL", "intruder@example.com")
    monkeypatch.setenv("STOCKOUT_ADMIN_PASSWORD", "intruder-password")

    with TestClient(create_app(), follow_redirects=False) as client:
        rejected = client.post(
            "/login", data={"email": "intruder@example.com", "password": "intruder-password"}
        )
        assert rejected.status_code == 401

    with db.session() as connection:
        assert users.count(connection) == 1


def test_an_unusable_seed_password_does_not_stop_the_app_booting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registration still works, and the first account made that way becomes the admin."""
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(tmp_path / "users.db"))
    monkeypatch.setenv("STOCKOUT_WEB_SECRET", "test-secret")
    monkeypatch.setenv("STOCKOUT_ADMIN_EMAIL", "seeded@example.com")
    monkeypatch.setenv("STOCKOUT_ADMIN_PASSWORD", "tiny")  # under the minimum length
    monkeypatch.setattr(ml_config, "ARTIFACT_DIR", tmp_path / "empty")

    with TestClient(create_app(), follow_redirects=False) as client:
        assert client.get("/health").status_code == 200
        created = client.post(
            "/register", data={"email": "somebody@example.com", "password": "a-real-password"}
        )
        assert created.status_code == 303
        assert client.get("/admin").status_code == 200


def test_no_seed_configured_leaves_an_empty_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is deliberately no built-in default account — a shipped admin/admin is a back door."""
    monkeypatch.setenv("STOCKOUT_WEB_DB", str(tmp_path / "users.db"))
    monkeypatch.setenv("STOCKOUT_WEB_SECRET", "test-secret")
    monkeypatch.delenv("STOCKOUT_ADMIN_EMAIL", raising=False)
    monkeypatch.delenv("STOCKOUT_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(ml_config, "ARTIFACT_DIR", tmp_path / "empty")

    with TestClient(create_app(), follow_redirects=False), db.session() as connection:
        assert users.count(connection) == 0


# --- the branches HTTP alone does not reach --------------------------------------------


def test_the_login_page_is_reachable_when_signed_out(client: TestClient) -> None:
    assert client.get("/login").status_code == 200


def test_the_registration_page_is_reachable(client: TestClient) -> None:
    assert client.get("/register").status_code == 200


def test_comparing_an_unregistered_model_is_refused(admin_client: TestClient) -> None:
    """A model name in a query string is untrusted input, like one in a form."""
    response = admin_client.get("/admin/comparison?task=regression&models=xgboost")
    assert response.status_code == 400
    assert "unknown regression model" in response.text


def test_forecasting_with_no_model_loaded_says_so(unloaded_client: TestClient) -> None:
    """The first-run state must produce a message, not a 500."""
    from .conftest import ADMIN, register

    register(unloaded_client, *ADMIN)
    response = unloaded_client.post(
        "/forecast", data={"store": "1", "date": "2015-01-02", "is_open": "true"}
    )
    assert response.status_code == 400
    assert "no model has been trained yet" in response.text
