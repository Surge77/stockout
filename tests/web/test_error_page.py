"""The error page has to actually say what went wrong.

`error.html` reads `code` and `message`. `main.py` was handing it `status` and `detail`,
and Jinja renders an undefined name as the empty string rather than raising — so every
403, 404 and refused domain error rendered a page with an empty heading and an empty red
box. It looked like a styling bug and was a broken context contract.

These tests pin the contract from the outside: whatever the keys are called, the status
code and the reason must appear in the HTML a user is shown.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from stockout_web.auth import current_user
from stockout_web.main import app
from stockout_web.users import User

PLAIN_USER = User(id=2, email="plain@example.com", role="user", is_active=True, created_at="now")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A signed-in non-admin. Overridden rather than logged in, so no database is needed.

    Module-scoped because entering `TestClient` runs the app's lifespan, and that fits
    every registered regressor — about twenty seconds. Per-test would pay it four times
    over for a set of assertions that never touch a model.
    """
    app.dependency_overrides[current_user] = lambda: PLAIN_USER
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_a_forbidden_page_shows_the_status_code(client: TestClient) -> None:
    response = client.get("/admin")

    assert response.status_code == 403
    assert "403" in response.text


def test_a_forbidden_page_shows_the_reason(client: TestClient) -> None:
    """The detail from the `HTTPException`, not an empty box."""
    response = client.get("/admin")

    assert "this page is for admins" in response.text


def test_a_missing_page_shows_both(client: TestClient) -> None:
    response = client.get("/no-such-page")

    assert response.status_code == 404
    assert "404" in response.text
    assert "Not Found" in response.text


def test_the_error_box_is_never_empty(client: TestClient) -> None:
    """The specific regression: an `alert error` element containing nothing at all."""
    body = client.get("/admin").text

    assert 'alert error">' in body, "the error page did not render its alert box"
    rendered = body.split('alert error">')[1].split("</p>")[0].strip()
    assert rendered, "the alert box rendered empty — the context keys do not match the template"
