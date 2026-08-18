"""The user module: one store-day in, two model answers out.

The assertion these are really about is that **both** halves come back. `stockout` solves
a regression problem and a classification problem on the same rows, and a page that
answered one while implying it had answered both would be the single most misleading thing
this app could do.
"""

from __future__ import annotations

import httpx2 as httpx
from fastapi.testclient import TestClient

from stockout.config import DEMAND_CLASS_LABELS


def _ask(client: TestClient, **overrides: object) -> httpx.Response:
    """One forecast request. Everything is a string, because an HTML form sends strings."""
    form = {"store": "1", "is_open": "true"}
    form.update({key: str(value) for key, value in overrides.items()})
    return client.post("/forecast", data=form)


def test_the_form_offers_only_stores_the_model_knows(member_client: TestClient) -> None:
    page = member_client.get("/").text
    assert 'name="store"' in page
    assert "<option value=\"1\"" in page


def test_a_forecast_returns_both_the_number_and_the_class(
    member_client: TestClient, answerable_date: str
) -> None:
    response = _ask(member_client, date=answerable_date)
    assert response.status_code == 200

    body = response.text
    assert "Regression &mdash; predicted sales" in body or "Regression" in body
    assert "Classification" in body
    assert any(label in body for label in DEMAND_CLASS_LABELS)


def test_the_answer_shows_the_cut_points_the_class_was_decided_by(
    member_client: TestClient, answerable_date: str
) -> None:
    """A label without its thresholds cannot be checked by the person reading it."""
    body = _ask(member_client, date=answerable_date).text
    assert "cut points" in body
    assert "Medium" in body


def test_a_promotion_changes_the_prediction(
    member_client: TestClient, answerable_date: str
) -> None:
    """Otherwise the future-known covariates are decoration on the form."""
    quiet = _ask(member_client, date=answerable_date).text
    promoted = _ask(member_client, date=answerable_date, promo="true").text
    assert quiet != promoted


def test_a_closed_store_is_answered_rather_than_refused(
    member_client: TestClient, answerable_date: str
) -> None:
    """A shut shop sells zero, and the trading calendar is known in advance."""
    response = member_client.post(
        "/forecast", data={"store": "1", "date": answerable_date}  # is_open omitted -> False
    )
    assert response.status_code == 200


def test_forecasting_past_the_horizon_is_refused_with_the_reason(
    member_client: TestClient, unanswerable_date: str
) -> None:
    """Beyond `last_date + horizon` the lag would be a prediction of a prediction."""
    response = _ask(member_client, date=unanswerable_date)
    assert response.status_code == 400
    assert "cannot forecast" in response.text


def test_a_store_outside_the_training_data_is_refused(
    member_client: TestClient, answerable_date: str
) -> None:
    response = _ask(member_client, store="99999", date=answerable_date)
    assert response.status_code == 400
    assert "not in the training data" in response.text


def test_a_missing_date_names_the_field_rather_than_five_hundred(
    member_client: TestClient,
) -> None:
    response = member_client.post("/forecast", data={"store": "1"})
    assert response.status_code == 422
    assert "date" in response.text


def test_forecasting_requires_an_account(client: TestClient, answerable_date: str) -> None:
    response = _ask(client, date=answerable_date)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_the_page_names_which_model_answered(
    member_client: TestClient, answerable_date: str
) -> None:
    """A served prediction with no way to ask "which model said this" is not auditable."""
    body = _ask(member_client, date=answerable_date).text
    assert "Which model said this" in body
    assert "trained" in body
