"""The user module: accounts, roles, and the only place a password is ever handled.

Three rules, and each exists because the obvious shortcut is a real vulnerability.

**A password is never stored, logged or returned.** `bcrypt` hashes it on the way in and
compares on the way back; nothing else in this package ever sees the plaintext, and
`User` deliberately has no field it could live in. The `password_hash` column is read
inside `verify` and nowhere else.

**Verification is constant-time, including for a user that does not exist.** Returning
early on an unknown email makes login measurably faster for addresses that are not
registered, which turns the login form into an account-enumeration oracle. `authenticate`
hashes against a dummy instead, so both paths cost the same.

**Roles are a closed set enforced twice** — once by the `CHECK` constraint in the schema
and once here. A role arriving from a form is data, and data does not get to invent
permissions.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import bcrypt

Role = Literal["user", "admin"]
ROLES: tuple[Role, ...] = ("user", "admin")

#: Hashed once at import and compared against when the email is unknown, so that a failed
#: login costs the same whether the account exists or not. bcrypt's own cost parameter is
#: what makes that expensive enough to matter.
_DUMMY_HASH = bcrypt.hashpw(b"no-such-user", bcrypt.gensalt())

#: bcrypt truncates silently at 72 bytes: two passwords sharing their first 72 characters
#: are the same password to it. Refused rather than truncated, because a silent truncation
#: means a user's last characters do nothing and nothing says so.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


class UserError(Exception):
    """A rejected account operation, with a message safe to show a user."""


@dataclass(frozen=True)
class User:
    """An account, without its password hash. Nothing here is a secret."""

    id: int
    email: str
    role: Role
    is_active: bool
    created_at: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def hash_password(password: str) -> str:
    """Validate, then hash. The only function in this package that makes a hash."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(f"a password needs at least {MIN_PASSWORD_LENGTH} characters")
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise UserError(f"a password may be at most {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("ascii")


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        email=str(row["email"]),
        role=str(row["role"]),  # type: ignore[arg-type]
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
    )


def create(
    connection: sqlite3.Connection, *, email: str, password: str, role: Role = "user"
) -> User:
    """Add an account. Raises `UserError` on a duplicate or an invalid role."""
    address = normalise_email(email)
    if role not in ROLES:
        raise UserError(f"unknown role {role!r}")

    digest = hash_password(password)
    try:
        cursor = connection.execute(
            "INSERT INTO users (email, password_hash, role, is_active, created_at)"
            " VALUES (?, ?, ?, 1, ?)",
            (address, digest, role, datetime.now(UTC).isoformat(timespec="seconds")),
        )
    except sqlite3.IntegrityError as exc:
        raise UserError(f"{address} already has an account") from exc

    created = by_id(connection, int(cursor.lastrowid or 0))
    if created is None:  # pragma: no cover - the row was just inserted
        raise UserError("the account could not be read back after being created")
    return created


def authenticate(connection: sqlite3.Connection, *, email: str, password: str) -> User | None:
    """The account for these credentials, or None. Never says *which* half was wrong.

    "No such user" and "wrong password" are one message to the caller on purpose:
    distinguishing them tells an attacker which addresses are registered.
    """
    row = connection.execute(
        "SELECT * FROM users WHERE email = ?", (normalise_email(email),)
    ).fetchone()

    supplied = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    if row is None:
        bcrypt.checkpw(supplied, _DUMMY_HASH)  # keep the timing indistinguishable
        return None

    if not bcrypt.checkpw(supplied, str(row["password_hash"]).encode("ascii")):
        return None
    if not bool(row["is_active"]):
        return None
    return _row_to_user(row)


def by_id(connection: sqlite3.Connection, user_id: int) -> User | None:
    row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row) if row else None


def listing(connection: sqlite3.Connection) -> list[User]:
    """Every account, oldest first. Admins are listed with everybody else."""
    rows = connection.execute("SELECT * FROM users ORDER BY id").fetchall()
    return [_row_to_user(row) for row in rows]


def count(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


def set_role(connection: sqlite3.Connection, *, user_id: int, role: Role) -> User:
    """Promote or demote. Refuses to remove the last admin."""
    if role not in ROLES:
        raise UserError(f"unknown role {role!r}")

    target = _require(connection, user_id)
    if target.is_admin and role != "admin":
        _require_another_admin(connection, besides=user_id)

    connection.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    return _require(connection, user_id)


def set_active(connection: sqlite3.Connection, *, user_id: int, is_active: bool) -> User:
    """Deactivate or restore. Refuses to deactivate the last admin.

    Deactivation rather than deletion: an account that has used the system is a fact about
    what happened, and `DELETE` makes an audit trail point at a row that is not there.
    """
    target = _require(connection, user_id)
    if target.is_admin and not is_active:
        _require_another_admin(connection, besides=user_id)

    connection.execute(
        "UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id)
    )
    return _require(connection, user_id)


def normalise_email(email: str) -> str:
    """Trimmed, lowercased, and sanity-checked. Not RFC 5322 validation, deliberately.

    Fully validating an email address in a regex is a famous way to reject real ones, and
    the only proof an address works is sending to it. What this rejects is input that
    cannot be an address at all: no `@`, more than one, nothing either side of it, or
    embedded whitespace. `"@"` used to pass, because the check was `"@" in address`.

    Lowercased because the column is `COLLATE NOCASE` — this keeps what is stored matching
    what is compared, rather than relying on the collation alone.
    """
    address = email.strip().lower()
    local, separator, domain = address.partition("@")
    unusable = (
        not separator
        or not local
        or not domain
        or "@" in domain
        or any(character.isspace() for character in address)
    )
    if unusable:
        raise UserError("that does not look like an email address")
    return address


def _require(connection: sqlite3.Connection, user_id: int) -> User:
    user = by_id(connection, user_id)
    if user is None:
        raise UserError("no such account")
    return user


def _require_another_admin(connection: sqlite3.Connection, *, besides: int) -> None:
    """Guard the last admin.

    Without it an admin can demote or deactivate themselves and lock everybody out of the
    admin module permanently, with no route back that does not involve editing the
    database by hand.
    """
    remaining = connection.execute(
        "SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND is_active = 1 AND id != ?",
        (besides,),
    ).fetchone()["n"]
    if int(remaining) == 0:
        raise UserError("this is the last active admin; promote somebody else first")
