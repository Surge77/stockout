"""The users table, in SQLite, with no ORM.

One table and five queries do not justify SQLAlchemy — its own dependency, its own
migration tool and its own mental model, to save about thirty lines here. `sqlite3` is in
the standard library, and this package's rule is not to add a dependency for something the
standard library already does.

**Every query below is parameterised.** `?` placeholders, never an f-string, without
exception — including the ones whose inputs "obviously cannot" be hostile. A query built
by concatenation is a habit rather than a decision, and the one that eventually takes user
input looks exactly like the ones that do not.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL CHECK (role IN ('user', 'admin')),
    is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at    TEXT    NOT NULL
);
"""

#: `COLLATE NOCASE` on the email is load-bearing rather than tidy. Without it
#: `Admin@x.com` and `admin@x.com` are two accounts, and a user who signs up with one
#: capitalisation can never log in with the other — a support ticket that looks like a
#: password problem and is not.


def connect(path: Path | None = None) -> sqlite3.Connection:
    """A connection with rows that behave like mappings and foreign keys enforced."""
    destination = Path(path) if path is not None else config.database_path()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # `check_same_thread=False` because Starlette serves requests from a thread pool and
    # the connection is opened per call anyway — see `session()` below, which never shares
    # one across requests.
    connection = sqlite3.connect(destination, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialise(path: Path | None = None) -> None:
    """Create the table if it is absent. Safe to call on every startup."""
    with session(path) as connection:
        connection.executescript(SCHEMA)


@contextmanager
def session(path: Path | None = None) -> Generator[sqlite3.Connection]:
    """One connection per unit of work, committed on success and rolled back on error.

    A connection per request rather than a shared one, because SQLite locks per writer
    and a long-lived connection held across an await is how a single slow request blocks
    every other. Opening one costs microseconds against a local file.
    """
    connection = connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
