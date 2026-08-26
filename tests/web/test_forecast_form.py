"""The day-kind selector, which used to be a yes/no box.

Rossmann distinguishes three kinds of public holiday — `a` ordinary, `b` Easter,
`c` Christmas — and the fitted models carry the distinction. A checkbox mapped all
three onto `a`, so the busiest and the deadest trading days of the year arrived at the
model wearing the same label and there was no way for a visitor to say which they meant.

These tests assert the three things a reader would check by eye: that the form offers
every kind, that the chosen one reaches the model unchanged, and that a code the form
never offered is refused rather than quietly read as an ordinary day.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stockout.errors import StockoutError

GOOD_PASSWORD = "correct-horse-battery"


def _signed_in(client: TestClient) -> None:
    client.post(
        "/register",
        data={"email": "boss@example.com", "password": GOOD_PASSWORD},
        follow_redirects=False,
    )


def _recording(calls: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """Record what the route passed the model, then stop.

    The route is the thing under test, not the model behind it, and the stub service
    cannot predict. Raising `StockoutError` after recording sends the page down its
    error branch, which renders — so the assertion can be made on what was recorded
    rather than on a prediction no stub could produce.

    Two seams, because the two routes reach their models differently. The forecast goes
    through `ModelService`, so the instance is patched. The described shop calls
    `described.describe` directly — a module function, like `catalogue.actual_for` — so
    the name the router imported is patched instead, through `monkeypatch` so it is
    restored rather than leaking into the next test.
    """

    def recorded(*_: object, **kwargs: object):
        calls.update(kwargs)
        raise StockoutError("recorded")

    from stockout_web import main
    from stockout_web.routers import forecast as forecast_router

    main.app.state.models.forecast = recorded
    monkeypatch.setattr(forecast_router, "describe", recorded)


@pytest.mark.parametrize("page", ["/", "/what-if"])
def test_the_form_offers_every_kind_of_holiday(client: TestClient, page: str) -> None:
    _signed_in(client)

    body = client.get(page).text

    for code in ('value="0"', 'value="a"', 'value="b"', 'value="c"'):
        assert code in body
    assert "Easter" in body
    assert "Christmas" in body


def test_the_chosen_kind_reaches_the_forecast_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: every kind used to arrive as `a`."""
    _signed_in(client)
    calls: dict[str, object] = {}
    _recording(calls, monkeypatch)

    client.post(
        "/forecast",
        data={"store": "1", "date": "2015-12-25", "state_holiday": "c"},
    )

    assert calls["state_holiday"] == "c"


def test_the_chosen_kind_reaches_the_described_shop_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _signed_in(client)
    calls: dict[str, object] = {}
    _recording(calls, monkeypatch)

    client.post(
        "/what-if",
        data={
            "store_type": "a",
            "assortment": "a",
            "date": "2015-12-25",
            "state_holiday": "b",
        },
    )

    assert calls["state_holiday"] == "b"


def test_an_ordinary_day_still_reaches_the_model_as_zero(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _signed_in(client)
    calls: dict[str, object] = {}
    _recording(calls, monkeypatch)

    client.post("/forecast", data={"store": "1", "date": "2015-06-01"})

    assert calls["state_holiday"] == "0"


def test_a_kind_the_form_never_offered_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A select is a hint to a browser, not a guarantee about what arrives."""
    _signed_in(client)
    calls: dict[str, object] = {}
    _recording(calls, monkeypatch)

    body = client.post(
        "/forecast",
        data={"store": "1", "date": "2015-06-01", "state_holiday": "z"},
    ).text

    # The code itself is quoted back with `repr`, which Jinja escapes on the way out —
    # so the assertion is on the sentence, not on the entity-encoded quotes around it.
    assert "not a kind of day" in body
    assert calls == {}
