"""The comparison page, which used to carry a column about the syllabus.

`evaluate.compare` reports a `taught` column saying which estimators come from the course
notebooks. That is a fact about how this project was built, and `scripts/check_taught.py`
still enforces it — but the page asks one question, *which model predicts sales best*, and
where an estimator was learnt has no bearing on the answer.

The column appeared twice on the page: as a table cell, and again inside the markdown
block underneath, which renders every column of the frame. Removing one and not the other
would have left it on screen.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

GOOD_PASSWORD = "correct-horse-battery"
COMPARISON = "/admin/comparison"


def _as_admin(client: TestClient) -> None:
    """The first account created is the administrator."""
    client.post(
        "/register",
        data={"email": "boss@example.com", "password": GOOD_PASSWORD},
        follow_redirects=False,
    )


def test_the_page_still_shows_every_model_and_its_score(client: TestClient) -> None:
    """The guard on the removal: a column went, the table did not."""
    _as_admin(client)

    body = client.get(COMPARISON).text

    assert "linear" in body
    assert "decision tree" in body
    assert "0.8296" in body


def test_the_course_column_is_gone_from_the_table(client: TestClient) -> None:
    _as_admin(client)

    body = client.get(COMPARISON).text

    assert "In the course material" not in body


def test_the_course_column_is_gone_from_the_markdown_too(client: TestClient) -> None:
    """The markdown block prints whatever columns the frame has."""
    _as_admin(client)

    body = client.get(COMPARISON).text

    assert "taught" not in body
