"""Who is asking, and may they. The dependencies every protected route hangs off.

The session is a signed cookie holding one integer — the user's id — and nothing else.
Not the role, and not the email. Both of those are read from the database on every
request, so that deactivating an account or demoting an admin takes effect immediately
rather than whenever their cookie happens to expire. A role copied into a cookie is a
permission the server can no longer revoke.

`current_user` and `require_admin` are the only two ways a route learns who it is serving.
A handler that checks `request.session` itself is a handler that will eventually forget
to, which is why nothing below returns the raw session to a caller.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from . import db, users
from .users import User

SESSION_KEY = "user_id"

#: A one-shot message that survives exactly one redirect. In the session rather than the
#: query string, because a message in a URL gets bookmarked, shared and replayed long
#: after it stopped being true.
NOTICE_KEY = "notice"


def set_notice(request: Request, message: str) -> None:
    """Leave a message for the page this request is about to redirect to."""
    request.session[NOTICE_KEY] = message


def take_notice(request: Request) -> str | None:
    """Read the pending message and clear it, so it shows once and not again."""
    return request.session.pop(NOTICE_KEY, None)


def login_session(request: Request, user: User) -> None:
    """Mark this browser as signed in as `user`.

    The session is cleared first so that logging in as somebody else cannot inherit
    anything the previous session left behind — session fixation, in one line.
    """
    request.session.clear()
    request.session[SESSION_KEY] = user.id
    _forget(request)


def logout_session(request: Request) -> None:
    request.session.clear()
    _forget(request)


def _forget(request: Request) -> None:
    """Drop the per-request cache after the session changes underneath it.

    Both callers currently redirect rather than render, so nothing would read a stale
    value today. That is a property of the current routes, not of this function, and it
    is not one worth relying on.
    """
    if hasattr(request.state, _REQUEST_CACHE):
        delattr(request.state, _REQUEST_CACHE)


#: Where the resolved account is cached for the life of one request.
_REQUEST_CACHE = "stockout_user"


def signed_in_user(request: Request) -> User | None:
    """The account this request belongs to, or None. Never raises.

    Used by the pages that render differently for a visitor than for a member. Routes that
    *require* an account use `current_user` instead, which turns absence into a redirect.

    Memoised on `request.state` because a protected page resolved the same account at
    least twice: once for the `CurrentUser` dependency and once inside `page`, which puts
    the user in every template's context. Each of those opened its own SQLite connection
    for the same row. The cache lives and dies with the request, so a role changed by an
    admin still takes effect on the very next one — which is the property the module
    docstring is about, and the reason this is not cached any longer than a request.
    """
    cached = getattr(request.state, _REQUEST_CACHE, _MISSING)
    if cached is not _MISSING:
        return cached  # type: ignore[return-value]

    user = _resolve(request)
    setattr(request.state, _REQUEST_CACHE, user)
    return user


class _Missing:
    """A sentinel, because `None` is a real cached answer: "nobody is signed in"."""


_MISSING = _Missing()


def _resolve(request: Request) -> User | None:
    user_id = request.session.get(SESSION_KEY)
    if not isinstance(user_id, int):
        return None

    with db.session() as connection:
        user = users.by_id(connection, user_id)

    if user is None or not user.is_active:
        # The account was deleted or deactivated while the cookie was still valid. Drop
        # the stale session rather than letting it keep resolving to nothing on every
        # subsequent request.
        request.session.clear()
        return None
    return user


def current_user(request: Request) -> User:
    """The signed-in account, or a 401. The dependency every member route uses."""
    user = signed_in_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="sign in to continue",
            headers={"Location": "/login"},
        )
    return user


def require_admin(user: Annotated[User, Depends(current_user)]) -> User:
    """The signed-in account, if it is an admin. 403 otherwise.

    403 rather than 404: hiding the existence of an admin page from somebody who already
    has an account buys nothing, and a wrong status code is a debugging session for
    whoever gets it.
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="this page is for admins"
        )
    return user


CurrentUser = Annotated[User, Depends(current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
