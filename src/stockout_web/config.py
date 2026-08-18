"""Settings for the web app, read from the environment at call time.

A separate package from `stockout` on purpose. `docs/architecture.md` claims that
everything between `data/` and `persistence.py` is pure — same input, same output, no
I/O — and a web app is nothing but I/O. Keeping it outside `src/stockout/` makes the
dependency direction unambiguous: the app imports the library, and the library has never
heard of the app.

Read lazily rather than captured into module constants at import, so a test can
`monkeypatch.setenv` without reimporting.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

#: Where the user table lives. SQLite, in the gitignored artifacts directory, because it
#: is build output rather than source — the same reasoning as the model file beside it.
DEFAULT_DATABASE = _ROOT / "artifacts" / "users.db"

#: How long a login lasts. A session that never expires is a credential with no end,
#: which is the one thing a cookie must not be.
SESSION_MAX_AGE_SECONDS = 60 * 60 * 8

#: Only these origins may call the API from a browser. Never `*`: this app runs on a
#: laptop and has no business accepting a cross-site request from anywhere else.
ALLOWED_ORIGINS: tuple[str, ...] = ("http://127.0.0.1:8000", "http://localhost:8000")

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def database_path() -> Path:
    """`STOCKOUT_WEB_DB`, or the default beside the model artifact."""
    override = os.getenv("STOCKOUT_WEB_DB", "").strip()
    return Path(override) if override else DEFAULT_DATABASE


def session_secret() -> str:
    """The key the session cookie is signed with.

    Generated per process when unset, which is right for development and wrong for
    anything else: every restart invalidates every session, and two workers would sign
    with different keys and reject each other's cookies. `STOCKOUT_WEB_SECRET` is how you
    fix that, and `.env.example` says so.

    Generated rather than defaulted to a constant, because a hard-coded fallback secret
    is the one that ends up in production — it works, so nobody notices it is public.
    """
    return os.getenv("STOCKOUT_WEB_SECRET", "").strip() or secrets.token_urlsafe(32)


def secure_cookies() -> bool:
    """Set `STOCKOUT_WEB_HTTPS=1` behind TLS so the session cookie is HTTPS-only.

    Off by default because the default is `http://127.0.0.1`, and a `Secure` cookie is
    never sent over plain HTTP — turning it on by default would make login silently fail
    on the one setup this app is actually run on.
    """
    return os.getenv("STOCKOUT_WEB_HTTPS", "").strip() in {"1", "true", "yes"}


def seed_admin() -> tuple[str, str] | None:
    """The first admin, from `STOCKOUT_ADMIN_EMAIL` and `STOCKOUT_ADMIN_PASSWORD`.

    Returns None when either is unset, and the app then starts with an empty user table
    and says so. There is deliberately no built-in default account: a shipped
    `admin/admin` is a back door, not a convenience.
    """
    email = os.getenv("STOCKOUT_ADMIN_EMAIL", "").strip()
    password = os.getenv("STOCKOUT_ADMIN_PASSWORD", "")
    return (email, password) if email and password else None
