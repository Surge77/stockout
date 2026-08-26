"""The row of "what every model said", and the two ways it drifted from the headline.

The forecast page prints one prediction large and lists every other regressor's answer
underneath. Those numbers are only comparable if they are the same *kind* of number, and
twice they were not.

**Units.** On a de-trended split every regressor answers in residuals — distances from the
store's own line. The headline puts the line back; the list did not, so a figure near zero
sat beside a figure near five thousand and read as a broken model rather than as a missing
addition. `Verdict.spread_low`/`spread_high` are taken from that list, so the range printed
under the headline was drawn from the wrong scale entirely.

**The question.** `state_holiday` and `school_holiday` were not passed down, so the list
answered an ordinary Tuesday while the headline answered the Christmas the visitor had
actually asked about. Two different questions, displayed as two disagreeing models.
"""

from __future__ import annotations

from typing import Any

import pytest

from stockout.datasets import prepare
from stockout.predict import compare_row, predict
from stockout.train import train

ENTITY = "1"
DATE = "2015-06-01"


@pytest.fixture(scope="module")
def prepared():
    return prepare("synthetic")


@pytest.fixture(scope="module")
def artifact(prepared, tmp_path_factory):
    built, _ = train(prepared, path=tmp_path_factory.mktemp("artifacts") / "model.joblib")
    return built


@pytest.fixture(scope="module")
def deployed(artifact) -> dict[str, object]:
    """The one fitted regressor the artefact carries, shaped like the web service's dict."""
    return {artifact.regressor_name: artifact.regressor}


def test_it_answers_in_sales_rather_than_in_residuals(artifact, deployed) -> None:
    """The deployed model appears in both places, so the two must agree exactly."""
    listed = compare_row(artifact, deployed, entity=ENTITY, date=DATE)
    headline = predict(artifact, entity=ENTITY, date=DATE).sales

    assert listed[artifact.regressor_name] == pytest.approx(headline)


def test_the_answer_is_on_the_same_scale_as_the_store(artifact, deployed, prepared) -> None:
    """A residual leaking through lands near zero; a sales figure lands near the median.

    Stated as a loose band rather than an exact figure: the point is the order of
    magnitude, and pinning the prediction itself would make this a change-detector.
    """
    listed = compare_row(artifact, deployed, entity=ENTITY, date=DATE)
    frame = prepared.frame
    typical = float(frame[frame["entity"].astype("string") == ENTITY]["sales"].median())

    assert listed[artifact.regressor_name] > typical / 2


class _Echo:
    """A stand-in regressor that returns one feature instead of a prediction.

    The plumbing is what these tests are about: does the submitted field reach the matrix
    the compared models are scored on. Asserting on a *fitted* model's output would test
    the data instead — on the synthetic sample gradient boosting never splits on
    `state_holiday`, so it answers an identical number for Christmas and for a Tuesday and
    a green test would prove only that this sample has no Christmases worth learning.
    """

    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, matrix: Any) -> Any:
        return matrix[self.column].to_numpy()


#: Any registered regression name works; `compare_row` looks models up by registry name.
_SLOT = "linear"


def test_the_kind_of_day_reaches_every_compared_model(artifact) -> None:
    """Christmas and a Tuesday must not arrive at the compared models as the same row."""
    echo = {_SLOT: _Echo("state_holiday")}
    ordinary = compare_row(artifact, echo, entity=ENTITY, date=DATE, state_holiday="0")
    christmas = compare_row(artifact, echo, entity=ENTITY, date=DATE, state_holiday="c")

    assert ordinary[_SLOT] != christmas[_SLOT]


def test_a_school_holiday_reaches_every_compared_model(artifact) -> None:
    echo = {_SLOT: _Echo("school_holiday")}
    plain = compare_row(artifact, echo, entity=ENTITY, date=DATE, school_holiday=False)
    holiday = compare_row(artifact, echo, entity=ENTITY, date=DATE, school_holiday=True)

    assert plain[_SLOT] != holiday[_SLOT]


def test_a_promotion_reaches_every_compared_model(artifact) -> None:
    echo = {_SLOT: _Echo("promo")}
    plain = compare_row(artifact, echo, entity=ENTITY, date=DATE, promo=False)
    running = compare_row(artifact, echo, entity=ENTITY, date=DATE, promo=True)

    assert plain[_SLOT] != running[_SLOT]
