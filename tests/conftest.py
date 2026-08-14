"""Shared fixtures, and a hard stop on network access.

The autouse guard is the reason `pytest` is trustworthy offline. Without it a test that
accidentally reaches Kaggle passes on a laptop with a token and fails in CI, and the
failure looks like a flake rather than a bug. Tests that genuinely need the network must
say so with `@pytest.mark.integration`, which is excluded from the default run.
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

from stockout.data.synth import make_sales


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration"):
        return

    def guard(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "a unit test tried to open a socket; mark it @pytest.mark.integration "
            "or stub the boundary"
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    """Two years of synthetic sales across three stores.

    Long enough for five 42-day folds on top of a 365-day minimum training window,
    which is the layout the CLI defaults to.
    """
    return make_sales(n_stores=3, days=730, seed=11)


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
