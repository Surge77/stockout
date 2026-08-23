"""The accounts page, which rendered an empty table over a populated database.

Three separate context mismatches, all invisible because Jinja renders an undefined name
as the empty string rather than raising:

* the route passed `rows`, the template looped over `accounts` — so every account was
  hidden, including the administrator reading the page;
* nothing ever passed `roles`, so both role dropdowns rendered with no options at all and
  there was no way to pick one;
* the deactivate button posted `is_active`, the route required a field called `active` —
  a 422 on a button that looked fine.

None of it raised, nothing logged, and the page looked merely empty. These tests assert
what a reader would check by eye, which is the only kind of test that would have caught it.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from stockout_web import db, users

GOOD_PASSWORD = "correct-horse-battery"
ADMIN_USERS = "/admin/users"
OTHER_PASSWORD = "another-good-password"


def _as_admin(client: TestClient) -> None:
    """The first account created is the administrator."""
    client.post(
        "/register",
        data={"email": "boss@example.com", "password": GOOD_PASSWORD},
        follow_redirects=False,
    )


def _add_user(client: TestClient, email: str) -> int:
    client.post(
        ADMIN_USERS,
        data={"email": email, "password": OTHER_PASSWORD, "role": "user"},
        follow_redirects=False,
    )
    with db.session() as connection:
        account = users.by_email(connection, email)
    assert account is not None
    return account.id


def test_the_table_lists_every_account(client: TestClient) -> None:
    """The regression. An admin looking at the page must at least see themselves."""
    _as_admin(client)
    _add_user(client, "clerk@example.com")

    body = client.get(ADMIN_USERS).text

    assert "boss@example.com" in body
    assert "clerk@example.com" in body


def test_the_table_body_is_not_empty(client: TestClient) -> None:
    """Pinned separately from the addresses: the bug rendered headers and no rows, so a
    page that merely *mentions* an address somewhere would still have passed."""
    _as_admin(client)

    body = client.get(ADMIN_USERS).text
    rows = re.search(r"<tbody>(.*?)</tbody>", body, re.S)

    assert rows is not None
    assert "<tr" in rows.group(1)


def test_the_role_dropdowns_offer_both_roles(client: TestClient) -> None:
    _as_admin(client)

    body = client.get(ADMIN_USERS).text

    for role in users.ROLES:
        assert f'value="{role}"' in body, f"the role select has no {role} option"


def test_the_create_form_can_actually_create(client: TestClient) -> None:
    """An empty role dropdown meant the add-account form posted no role at all."""
    _as_admin(client)

    client.post(
        ADMIN_USERS,
        data={"email": "new@example.com", "password": OTHER_PASSWORD, "role": "admin"},
        follow_redirects=False,
    )

    with db.session() as connection:
        created = users.by_email(connection, "new@example.com")
    assert created is not None
    assert created.role == "admin"


def test_deactivating_from_the_page_works(client: TestClient) -> None:
    """The button posted a field the route did not accept, so it answered 422."""
    _as_admin(client)
    victim = _add_user(client, "clerk@example.com")

    response = client.post(
        f"{ADMIN_USERS}/{victim}/active", data={"is_active": "false"}, follow_redirects=False
    )

    assert response.status_code == 200, f"the deactivate button answered {response.status_code}"
    with db.session() as connection:
        account = users.by_id(connection, victim)
    assert account is not None and not account.is_active


def test_restoring_from_the_page_works(client: TestClient) -> None:
    _as_admin(client)
    victim = _add_user(client, "clerk@example.com")

    client.post(f"{ADMIN_USERS}/{victim}/active", data={"is_active": "false"})
    client.post(f"{ADMIN_USERS}/{victim}/active", data={"is_active": "true"})

    with db.session() as connection:
        account = users.by_id(connection, victim)
    assert account is not None and account.is_active


def test_changing_a_role_from_the_page_works(client: TestClient) -> None:
    _as_admin(client)
    target = _add_user(client, "clerk@example.com")

    client.post(f"{ADMIN_USERS}/{target}/role", data={"role": "admin"})

    with db.session() as connection:
        account = users.by_id(connection, target)
    assert account is not None and account.role == "admin"


def test_the_table_still_lists_accounts_after_a_change(client: TestClient) -> None:
    """`_mutate` re-renders the same template, and had the same wrong context key."""
    _as_admin(client)
    target = _add_user(client, "clerk@example.com")

    body = client.post(f"{ADMIN_USERS}/{target}/role", data={"role": "admin"}).text

    assert "clerk@example.com" in body


def test_a_refused_change_is_shown_and_keeps_the_table(client: TestClient) -> None:
    """Demoting the only admin must fail loudly, with the list still on the page."""
    _as_admin(client)
    with db.session() as connection:
        boss = users.by_email(connection, "boss@example.com")
    assert boss is not None

    body = client.post(f"{ADMIN_USERS}/{boss.id}/role", data={"role": "user"}).text

    assert "last active admin" in body
    assert "boss@example.com" in body
