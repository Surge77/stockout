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


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> HTMLResponse:
    if auth.signed_in_user(request) is not None:
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]
    return page(request, "login.html", {"mode": "login"})


@router.post("/login", response_class=HTMLResponse)
async def login(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> HTMLResponse:
    with db.session() as connection:
        user = users.authenticate(connection, email=email, password=password)

    if user is None:
        # 401 with the form re-rendered, rather than a redirect carrying the error in a
        # query string — an error in a URL survives being shared and bookmarked.
        return page(
            request,
            "login.html",
            {"mode": "login", "error": _REJECTED, "email": email},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    auth.login_session(request, user)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]


@router.get("/register", response_class=HTMLResponse)
async def register_form(request: Request) -> HTMLResponse:
    return page(request, "login.html", {"mode": "register"})


@router.post("/register", response_class=HTMLResponse)
async def register(
    request: Request,
    email: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> HTMLResponse:
    """Create a `user` account. Never an admin — see the module docstring."""
    try:
        with db.session() as connection:
            first_account = users.count(connection) == 0
            # The very first account to exist becomes the admin, because otherwise a
            # fresh install with no seed credentials has no way to reach the admin
            # module at all. Every account after it is a plain user.
            user = users.create(
                connection,
                email=email,
                password=password,
                role="admin" if first_account else "user",
            )
    except users.UserError as exc:
        return page(
            request,
            "login.html",
            {"mode": "register", "error": str(exc), "email": email},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    auth.login_session(request, user)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)  # type: ignore[return-value]


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    """POST rather than GET, so a link on another site cannot sign somebody out."""
    auth.logout_session(request)
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
