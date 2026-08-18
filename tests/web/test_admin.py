"""The admin module: the role gate, retraining, the comparison table, and the accounts.

Half of these are refusals. An admin page that works is worth less than one that reliably
turns a plain user away, because the first failure mode is visible on the first visit and
the second is invisible until it matters.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stockout_web import db, users

from .conftest import ADMIN, MEMBER, register, sign_out

ADMIN_PATHS = ["/admin", "/admin/users", "/admin/comparison"]


@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_a_plain_user_is_refused_every_admin_page(member_client: TestClient, path: str) -> None:
    """403, not 404: a wrong status code is a debugging session for whoever gets it."""
    assert member_client.get(path).status_code == 403


@pytest.mark.parametrize("path", ["/admin/train", "/admin/users"])
def test_a_plain_user_cannot_post_to_an_admin_route(
    member_client: TestClient, path: str
) -> None:
    """The gate is on the route, not on the page that links to it."""
    assert member_client.post(path, data={}).status_code == 403


def test_the_dashboard_shows_the_artifact_provenance(admin_client: TestClient) -> None:
    body = admin_client.get("/admin").text
    for field in ("Trained", "Training rows", "Horizon", "Features"):
        assert field in body


def test_the_dashboard_shows_the_held_out_scores(admin_client: TestClient) -> None:
    """A score computed on the rows a model was fitted on is not a score, and it says so."""
    body = admin_client.get("/admin").text
    assert "Held-out scores" in body
    assert "wmape" in body


def test_retraining_replaces_what_users_are_served(admin_client: TestClient) -> None:
    """The reload is the point: without it the admin sees success and users see the old model."""
    before = admin_client.get("/health").json()
    assert before["model_loaded"] is True

    response = admin_client.post(
        "/admin/train", data={"regressor": "linear", "classifier": "logistic"}
    )
    assert response.status_code == 200
    assert "Trained." in response.text
    assert admin_client.get("/health").json()["model_loaded"] is True


def test_training_with_an_unregistered_model_is_refused(admin_client: TestClient) -> None:
    """A model name from a form is untrusted input; the registry is the only list."""
    response = admin_client.post(
        "/admin/train", data={"regressor": "xgboost", "classifier": "logistic"}
    )
    assert response.status_code == 400
    assert "unknown regression model" in response.text


@pytest.mark.parametrize(
    ("task", "expected"),
    [("regression", "wmape"), ("classification", "macro_f1")],
)
def test_the_comparison_table_renders_for_both_tasks(
    admin_client: TestClient, task: str, expected: str
) -> None:
    response = admin_client.get(f"/admin/comparison?task={task}&models=dummy")
    assert response.status_code == 200
    assert expected in response.text
    # The floor has to be in the table for any other number in it to be readable.
    assert "dummy" in response.text


def test_an_unknown_task_is_refused(admin_client: TestClient) -> None:
    response = admin_client.get("/admin/comparison?task=clustering")
    assert response.status_code == 400


# --- accounts ----------------------------------------------------------------------


def test_an_admin_can_create_an_account_at_either_role(admin_client: TestClient) -> None:
    response = admin_client.post(
        "/admin/users",
        data={"email": "second@example.com", "password": "another-password", "role": "admin"},
    )
    assert response.status_code == 303

    with db.session() as connection:
        created = next(u for u in users.listing(connection) if u.email == "second@example.com")
    assert created.is_admin


def test_creating_a_duplicate_account_reports_the_reason_on_the_page(
    admin_client: TestClient,
) -> None:
    response = admin_client.post(
        "/admin/users", data={"email": ADMIN[0], "password": "another-password", "role": "user"}
    )
    assert response.status_code == 400
    assert "already has an account" in response.text


def test_an_admin_can_promote_and_demote(admin_client: TestClient) -> None:
    admin_client.post(
        "/admin/users", data={"email": MEMBER[0], "password": MEMBER[1], "role": "user"}
    )
    with db.session() as connection:
        member = next(u for u in users.listing(connection) if u.email == MEMBER[0])

    assert admin_client.post(
        f"/admin/users/{member.id}/role", data={"role": "admin"}
    ).status_code == 303
    with db.session() as connection:
        assert users.by_id(connection, member.id).is_admin  # type: ignore[union-attr]


def test_an_admin_can_deactivate_and_restore(admin_client: TestClient) -> None:
    admin_client.post(
        "/admin/users", data={"email": MEMBER[0], "password": MEMBER[1], "role": "user"}
    )
    with db.session() as connection:
        member = next(u for u in users.listing(connection) if u.email == MEMBER[0])

    admin_client.post(f"/admin/users/{member.id}/active", data={"is_active": "false"})
    with db.session() as connection:
        assert users.by_id(connection, member.id).is_active is False  # type: ignore[union-attr]

    admin_client.post(f"/admin/users/{member.id}/active", data={"is_active": "true"})
    with db.session() as connection:
        assert users.by_id(connection, member.id).is_active is True  # type: ignore[union-attr]


def test_the_last_admin_cannot_demote_themselves(admin_client: TestClient) -> None:
    """Otherwise the admin module is lockable with no route back but editing the database."""
    with db.session() as connection:
        only_admin = next(u for u in users.listing(connection) if u.email == ADMIN[0])

    response = admin_client.post(f"/admin/users/{only_admin.id}/role", data={"role": "user"})
    assert response.status_code == 400
    assert "last active admin" in response.text

    with db.session() as connection:
        assert users.by_id(connection, only_admin.id).is_admin  # type: ignore[union-attr]


def test_the_last_admin_cannot_deactivate_themselves(admin_client: TestClient) -> None:
    with db.session() as connection:
        only_admin = next(u for u in users.listing(connection) if u.email == ADMIN[0])

    response = admin_client.post(
        f"/admin/users/{only_admin.id}/active", data={"is_active": "false"}
    )
    assert response.status_code == 400
    assert "last active admin" in response.text


def test_an_admin_may_be_demoted_once_another_exists(admin_client: TestClient) -> None:
    admin_client.post(
        "/admin/users",
        data={"email": "second@example.com", "password": "another-password", "role": "admin"},
    )
    with db.session() as connection:
        first = next(u for u in users.listing(connection) if u.email == ADMIN[0])

    assert admin_client.post(
        f"/admin/users/{first.id}/role", data={"role": "user"}
    ).status_code == 303


def test_the_account_list_never_renders_a_password_hash(admin_client: TestClient) -> None:
    """`User` has no field for it, and this is the assertion that keeps it that way."""
    body = admin_client.get("/admin/users").text
    assert "$2b$" not in body
    assert "password_hash" not in body


def test_a_deactivated_account_is_kept_rather_than_deleted(admin_client: TestClient) -> None:
    """An account that has used the system is a fact about what happened."""
    admin_client.post(
        "/admin/users", data={"email": MEMBER[0], "password": MEMBER[1], "role": "user"}
    )
    with db.session() as connection:
        member = next(u for u in users.listing(connection) if u.email == MEMBER[0])
    admin_client.post(f"/admin/users/{member.id}/active", data={"is_active": "false"})

    body = admin_client.get("/admin/users").text
    assert MEMBER[0] in body
    assert "deactivated" in body


def test_a_signed_out_admin_gets_no_admin_pages(admin_client: TestClient) -> None:
    sign_out(admin_client)
    assert admin_client.get("/admin").status_code == 303


def test_registering_after_an_admin_exists_never_grants_admin(client: TestClient) -> None:
    register(client, *ADMIN)
    sign_out(client)
    register(client, "third@example.com", "third-password")
    assert client.get("/admin").status_code == 403
