"""Sign in, sign out, and the boundaries that decide who sees what.

Auth is the one part of this app where a bug is a vulnerability rather than a wrong
number, so these test the *refusals* at least as hard as the successes: a wrong password,
a deactivated account, a plain user reaching for an admin page, and the message a failed
login is allowed to give away.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stockout_web import db, users

from .conftest import ADMIN, MEMBER, register, sign_in, sign_out


def test_the_first_account_becomes_the_admin(client: TestClient) -> None:
    """Otherwise a fresh install with no seed credentials can never reach /admin."""
    register(client, *ADMIN)
    assert client.get("/admin").status_code == 200


def test_every_account_after_the_first_is_a_plain_user(client: TestClient) -> None:
    """A signup form that could mint an admin would be an admin account for the asking."""
    register(client, *ADMIN)
    sign_out(client)
    register(client, *MEMBER)

    assert client.get("/admin").status_code == 403
    assert client.get("/").status_code == 200


def test_a_password_is_never_stored_in_the_clear(client: TestClient) -> None:
    register(client, *ADMIN)
    email, password = ADMIN

    with db.session() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()

    stored = str(row["password_hash"])
    assert password not in stored
    assert stored.startswith("$2b$")  # bcrypt, not a hex digest of something


def test_a_wrong_password_is_rejected(client: TestClient) -> None:
    register(client, *ADMIN)
    sign_out(client)
    response = client.post("/login", data={"email": ADMIN[0], "password": "not-the-password"})
    assert response.status_code == 401


def test_an_unknown_email_and_a_wrong_password_give_the_same_message(
    client: TestClient,
) -> None:
    """Distinguishing them turns the login form into an account-enumeration oracle."""
    register(client, *ADMIN)
    sign_out(client)

    wrong_password = client.post("/login", data={"email": ADMIN[0], "password": "wrong-one"})
    unknown_email = client.post(
        "/login", data={"email": "nobody@example.com", "password": "wrong-one"}
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert "That email or password is wrong." in wrong_password.text
    assert "That email or password is wrong." in unknown_email.text


def test_a_deactivated_account_cannot_sign_in(client: TestClient) -> None:
    register(client, *ADMIN)
    sign_out(client)
    register(client, *MEMBER)
    sign_out(client)

    with db.session() as connection:
        member = next(u for u in users.listing(connection) if u.email == MEMBER[0])
        users.set_active(connection, user_id=member.id, is_active=False)

    response = client.post("/login", data={"email": MEMBER[0], "password": MEMBER[1]})
    assert response.status_code == 401


def test_deactivating_a_signed_in_user_ends_their_session_immediately(
    member_client: TestClient,
) -> None:
    """The role and status are read per request, never trusted from the cookie."""
    assert member_client.get("/").status_code == 200

    with db.session() as connection:
        member = next(u for u in users.listing(connection) if u.email == MEMBER[0])
        users.set_active(connection, user_id=member.id, is_active=False)

    # The cookie is still valid and still signed; the account behind it is not.
    assert member_client.get("/").status_code == 303


def test_signing_out_ends_the_session(admin_client: TestClient) -> None:
    assert admin_client.get("/").status_code == 200
    sign_out(admin_client)
    assert admin_client.get("/").status_code == 303


def test_logout_refuses_a_get(admin_client: TestClient) -> None:
    """POST only, so a link or an image on another site cannot sign somebody out."""
    assert admin_client.get("/logout").status_code == 405


@pytest.mark.parametrize("path", ["/", "/admin", "/admin/users", "/admin/comparison"])
def test_a_signed_out_visitor_is_sent_to_the_login_page(
    client: TestClient, path: str
) -> None:
    response = client.get(path)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_a_signed_in_user_visiting_login_is_sent_home(admin_client: TestClient) -> None:
    response = admin_client.get("/login")
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_a_duplicate_email_is_refused_with_a_readable_message(client: TestClient) -> None:
    register(client, *ADMIN)
    sign_out(client)
    response = client.post("/register", data={"email": ADMIN[0], "password": "another-password"})
    assert response.status_code == 400
    assert "already has an account" in response.text


def test_a_short_password_is_refused(client: TestClient) -> None:
    response = client.post("/register", data={"email": "tiny@example.com", "password": "short"})
    assert response.status_code == 400
    assert "at least 8 characters" in response.text


def test_email_case_does_not_create_a_second_account(client: TestClient) -> None:
    """`Admin@x` and `admin@x` are one account, or a user can lock themselves out."""
    register(client, *ADMIN)
    sign_out(client)
    sign_in(client, ADMIN[0].upper(), ADMIN[1])
    assert client.get("/").status_code == 200


def test_the_health_check_needs_no_account(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_the_app_starts_and_serves_registration_with_no_model(
    unloaded_client: TestClient,
) -> None:
    """First run: an admin must be able to sign in *before* a model exists."""
    assert unloaded_client.get("/health").json()["model_loaded"] is False
    register(unloaded_client, *ADMIN)

    home = unloaded_client.get("/")
    assert home.status_code == 200
    assert "No model is loaded" in home.text
