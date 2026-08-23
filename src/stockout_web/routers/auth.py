"""Sign in, sign out, and the first-run account.

The one thing worth reading twice is what a failed login says: *"email or password is
wrong"*, never which. Telling somebody the email was fine narrows an attack from "guess a
pair" to "guess a password for an address I have now confirmed exists", and the friendlier
message is the one that leaks.

Registration is deliberately open, and every self-registered account is a plain `user`.
An admin can only be made by another admin, or by the seed credentials in the environment
on first run. A signup form that took a role would be an admin account for the asking.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import auth, db, users
from ..templating import page

router = APIRouter(tags=["auth"])

#: Deliberately identical for an unknown address and a wrong password.
_REJECTED = "That email or password is wrong."


#: Where each role belongs once it is signed in.
_USER_HOME = "/"
_ADMIN_HOME = "/admin"


def _home_for(user: users.User) -> str:
    return _ADMIN_HOME if user.is_admin else _USER_HOME


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> HTMLResponse:
    signed_in = auth.signed_in_user(request)
    if signed_in is not None:
        return RedirectResponse(_home_for(signed_in), status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
    return page(request, "login.html", {"mode": "login", "door": "user"})


@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login_form(request: Request) -> HTMLResponse:
    """The administrator door.

    A separate page rather than a role selector on one, because "which door did you come
    through" is a statement of intent and nothing more. It cannot be allowed to influence
    what you get: the role is read from the database after the password checks out, so
    posting here with an ordinary account signs you in as that ordinary account. A form
    field that decided your permissions would be an admin account for the asking.

    Deliberately outside the admin router: every route there depends on `AdminUser`, and
    a sign-in page you must already be an admin to reach is a locked door with the key
    behind it.
    """
    signed_in = auth.signed_in_user(request)
    if signed_in is not None:
        return RedirectResponse(_home_for(signed_in), status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
    return page(request, "login.html", {"mode": "login", "door": "admin"})


@router.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> HTMLResponse:
    return _sign_in(request, email=email, password=password, door="user")


@router.post("/admin/login", response_class=HTMLResponse)
async def admin_login(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> HTMLResponse:
    return _sign_in(request, email=email, password=password, door="admin")


def _sign_in(request: Request, *, email: str, password: str, door: str) -> HTMLResponse:
    """One credential check behind both doors, and the same refusal text for both.

    The door chooses the wording and nothing else. In particular it does not change the
    failure message: if the admin form said "that account is not an administrator" while
    the user form said "wrong password", the pair of them would report which addresses
    hold admin — the enumeration hole the single message exists to close.
    """
    with db.session() as connection:
        user = users.authenticate(connection, email=email, password=password)

    if user is None:
        # 401 with the form re-rendered, rather than a redirect carrying the error in a
        # query string — an error in a URL survives being shared and bookmarked.
        return page(
            request,
            "login.html",
            {"mode": "login", "door": door, "error": _REJECTED, "email": email},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    auth.login_session(request, user)
    if door == "admin" and not user.is_admin:
        # Valid credentials, wrong door. Signing them in and saying so beats refusing a
        # correct password, and the notice explains the page they did not get.
        auth.set_notice(
            request,
            "Signed in as a standard user. The admin area needs an administrator account.",
        )
    return RedirectResponse(_home_for(user), status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request) -> HTMLResponse:
    """The form, told in advance which kind of account it is about to create.

    There is one sign-in form and no role to choose, so the only place the distinction can
    be explained is here, before the account exists. Saying "the first account becomes the
    admin" in the abstract leaves the reader working out which case they are in; saying
    *this one will be the administrator* does not.
    """
    if auth.signed_in_user(request) is not None:
        # Matching `login_form`. Without it a signed-in user can open this page and
        # register a second account, which silently replaces the session they were using.
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
    context = {"mode": "register", "is_first_account": _no_accounts_yet()}
    return page(request, "login.html", context)


def _no_accounts_yet() -> bool:
    """Whether the next registration would be the first, and so the administrator.

    Advisory only. The role is still decided inside the `INSERT` — this reads the count
    outside any transaction and two people racing would both be told "admin" while only
    one gets it. Correct for a label, never correct for the grant itself.
    """
    with db.session() as connection:
        return users.count(connection) == 0


@router.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> HTMLResponse:
    """Create a `user` account. Never an admin — see the module docstring."""
    try:
        with db.session() as connection:
            # The very first account to exist becomes the admin, because otherwise a
            # fresh install with no seed credentials has no way to reach the admin
            # module at all. Every account after it is a plain user — and the decision
            # is made *inside* the insert, because reading the count first lets two
            # simultaneous registrations both see an empty table and both become one.
            user = users.create_self_registered(
                connection, email=email, password=password
            )
    except users.UserError as exc:
        return page(
            request,
            "login.html",
            {
                "mode": "register",
                "error": str(exc),
                "email": email,
                "is_first_account": _no_accounts_yet(),
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    auth.login_session(request, user)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    """POST rather than GET, so a link on another site cannot sign somebody out."""
    auth.logout_session(request)
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
