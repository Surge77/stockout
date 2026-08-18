"""Shared fixtures, and a hard stop on network access.

The autouse guard is the reason `pytest` is trustworthy offline. Without it a test that
accidentally reaches Kaggle passes on a laptop with a token and fails in CI, and the
failure looks like a flake rather than a bug. Tests that genuinely need the network must
say so with `@pytest.mark.integration`, which is excluded from the default run.

**Loopback is allowed, and the distinction is the point.** The guard exists to stop a test
*leaving the machine*, not to stop it using a socket. Python's own event loop builds its
wake-up self-pipe with `socket.socketpair()`, which on Windows is a pair of connected
127.0.0.1 sockets — so a blanket refusal makes `TestClient` impossible to use while
blocking nothing a test could have reached. `tests/web/` drives the whole web app in
process through exactly that mechanism.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.loaders import write_sales
from stockout.data.synth import make_sales
from stockout.data.synth_stores import make_stores


def one_store_frame(sales: list[float], *, open_flags: list[int] | None = None) -> pd.DataFrame:
    """One store, consecutive days, with `sales` as given."""
    flags = [1] * len(sales) if open_flags is None else open_flags
    return pd.DataFrame(
        {
            s.DATE: pd.date_range("2024-01-01", periods=len(sales), freq="D"),
            s.STORE: 1,
            s.SALES: sales,
            s.OPEN: flags,
        }
    )


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration"):
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def is_loopback(address: object) -> bool:
        """True for 127.0.0.0/8, ::1 and a unix socket. Nothing here leaves the machine."""
        if not isinstance(address, tuple) or not address:
            return True  # AF_UNIX and friends take a path, not a host/port pair
        host = address[0]
        return isinstance(host, str) and (
            host in {"localhost", "::1", ""} or host.startswith("127.")
        )

    def refuse(address: object) -> None:
        raise RuntimeError(
            f"a unit test tried to reach {address!r}; mark it @pytest.mark.integration "
            "or stub the boundary"
        )

    def guarded_connect(self: socket.socket, address: object) -> object:
        if not is_loopback(address):
            refuse(address)
        return real_connect(self, address)  # type: ignore[arg-type]

    def guarded_connect_ex(self: socket.socket, address: object) -> object:
        if not is_loopback(address):
            refuse(address)
        return real_connect_ex(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    """Two years of synthetic sales across three stores.

    Long enough for five 42-day folds on top of a 365-day minimum training window,
    which is the layout the CLI defaults to.
    """
    return make_sales(n_stores=3, days=730, seed=11)


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    """A written CSV for the CLI tests, which take a path rather than a frame.

    Two stores on purpose: the smallest file that exercises per-store behaviour — the
    per-store demand terciles, the per-store lag construction and the baselines' own
    store-level fallbacks — while staying fast enough to run in every CLI test.
    """
    return write_sales(make_sales(n_stores=2, days=730, seed=13), tmp_path / "sales.csv")


@pytest.fixture
def store_file(tmp_path: Path) -> Path:
    """The metadata half of the pair, for the commands that call `dataset.prepare`.

    Written beside `data_file` and covering the same store ids. Every command that fits
    a model needs both files, because four of the features come from the join and
    `prepare` refuses to proceed without it rather than dropping them in silence.
    """
    return write_sales(make_stores(n_stores=2, seed=13), tmp_path / "stores.csv")


@pytest.fixture
def tiny() -> pd.DataFrame:
    """Twelve hand-checkable rows across two stores, for lag and rolling arithmetic."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    rows = []
    for store, base in ((1, 100.0), (2, 500.0)):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "store": store,
                    "day_of_week": date.dayofweek + 1,
                    "sales": base + offset,
                    "customers": 10.0,
                    "open": 1,
                    "promo": 0,
                    "state_holiday": "0",
                    "school_holiday": 0,
                }
            )
    return pd.DataFrame(rows)
