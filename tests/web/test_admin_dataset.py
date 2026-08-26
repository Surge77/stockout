"""Deploying a different dataset from the dashboard, which is what makes the product
pages reachable without the CLI.

`/admin/train` could only ever refit the *same* data with a different pair — it passed
`dataset=service.artifact.dataset` and there was no field for anything else. So the whole
product half of this project, the shelf page included, was unreachable from the website:
the only way to see it was `python -m stockout train --dataset synthetic_products` at a
terminal.

The retrain itself is covered by `tests/test_products.py` against real fits. These tests
are about the route: that the field is offered, that it is validated, and that it reaches
the service.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from stockout_web.users import User

GOOD_PASSWORD = "correct-horse-battery"
ADMIN = User(id=1, email="boss@example.com", role="admin", is_active=True, created_at="now")


class _Recorder:
    """A stand-in for the retrain, recording what the route asked for."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def __call__(self, **kwargs: str):
        self.calls.append(kwargs)
        from stockout_web import main

        return main.app.state.models


@pytest.fixture
def admin_client(client: TestClient) -> Iterator[TestClient]:
    """The shared stubbed app, with an admin signed in so `/admin` is reachable.

    Builds on `conftest.client` rather than standing up a second one — pytest lets a
    fixture request the same name from an outer scope — so this inherits the stubbed
    `ModelService` and never fits a model.
    """
    from stockout_web import main
    from stockout_web.auth import require_admin

    main.app.dependency_overrides[require_admin] = lambda: ADMIN
    yield client
    main.app.dependency_overrides.clear()


def test_the_dashboard_offers_the_datasets_on_disk(admin_client: TestClient) -> None:
    body = admin_client.get("/admin").text

    assert 'name="dataset"' in body
    assert "synthetic_products" in body
    assert "synthetic" in body


def test_it_says_that_changing_the_dataset_changes_the_pages(admin_client: TestClient) -> None:
    """The consequence is not obvious from a dropdown, and it is the reason to use it."""
    body = admin_client.get("/admin").text

    assert "changes which pages exist" in body


def test_the_chosen_dataset_reaches_the_retrain(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from stockout_web import main

    recorder = _Recorder()
    monkeypatch.setattr(main.app.state.models, "retrain", recorder, raising=False)

    admin_client.post(
        "/admin/train",
        data={
            "regressor": "linear",
            "classifier": "logistic",
            "dataset": "synthetic_products",
        },
    )

    assert recorder.calls == [
        {"regressor": "linear", "classifier": "logistic", "dataset": "synthetic_products"}
    ]


def test_omitting_the_field_keeps_the_deployed_dataset(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The field is new; a form without it must keep meaning "refit this data"."""
    from stockout_web import main

    recorder = _Recorder()
    monkeypatch.setattr(main.app.state.models, "retrain", recorder, raising=False)

    admin_client.post("/admin/train", data={"regressor": "linear", "classifier": "logistic"})

    assert recorder.calls[0]["dataset"] == "synthetic"


def test_a_dataset_that_is_not_offered_is_refused(
    admin_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `<select>` constrains a browser, not a request.

    Without the check an unknown name reaches `prepare` and surfaces as a stack trace
    rather than a refusal naming what is available.
    """
    from stockout_web import main

    recorder = _Recorder()
    monkeypatch.setattr(main.app.state.models, "retrain", recorder, raising=False)

    response = admin_client.post(
        "/admin/train",
        data={"regressor": "linear", "classifier": "logistic", "dataset": "nonesuch"},
    )

    assert response.status_code == 200
    assert "is not a deployable dataset" in response.text
    assert recorder.calls == []
