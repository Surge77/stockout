"""Serving a date past the end of the data, and the one feature family that breaks there.

Two things had to be true before the panel could answer a future date at all.

The guard had to move. It lived in `stockout_web.service`, so the CLI reached
`serving_row` without passing anything at all and would answer any date it could parse.
It now lives in `stockout.window` and is called from `serving_row`, which every surface
goes through.

The store ages had to be capped. This is the part that is easy to miss, and the reason
"just allow future dates" is not the whole change. The calendar features are cyclic —
`day_of_week`, `month`, `week_of_year` for a day in 2030 are values the fit saw thousands
of times — so they extrapolate for free and must keep the real date. The store ages are
monotone: `competition_open_age_months` grows without bound, so a date years past the data
hands the model a value beyond every split point it was built from, on a feature carrying
about a sixth of the deployed model's importance. Capped, the model reads it as *this
competitor has been open the longest I ever observed*, which is the honest reading.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockout import features
from stockout.data import schemas as s
from stockout.datasets import prepare
from stockout.errors import StockoutError
from stockout.predict import date_status, predict, serving_row
from stockout.train import train

CAP = pd.Timestamp("2015-12-31")


@pytest.fixture(scope="module")
def prepared():
    return prepare("synthetic")


@pytest.fixture(scope="module")
def artifact(prepared, tmp_path_factory):
    built, _ = train(prepared, path=tmp_path_factory.mktemp("artifacts") / "model.joblib")
    return built


# -- the age cap, at the level it is implemented -------------------------------


def _one_row(date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                s.ENTITY: "1",
                s.DATE: pd.Timestamp(date),
                s.SALES: 0.0,
                s.OPEN: 1,
                s.PROMO: 0,
                s.STATE_HOLIDAY: "0",
                s.SCHOOL_HOLIDAY: 0,
            }
        ]
    )


def _stores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                s.ENTITY: "1",
                s.STORE_TYPE: "a",
                s.ASSORTMENT: "a",
                s.COMPETITION_DISTANCE: 1_000.0,
                s.COMPETITION_OPEN_YEAR: 2010.0,
                s.COMPETITION_OPEN_MONTH: 1.0,
                s.PROMO2: 0,
                s.PROMO2_SINCE_YEAR: float("nan"),
                s.PROMO2_SINCE_WEEK: float("nan"),
                s.PROMO_INTERVAL: "",
            }
        ]
    )


def test_an_uncapped_future_date_runs_the_age_far_past_training() -> None:
    """The failure the cap prevents, measured rather than asserted."""
    at_cap, _ = features.build(_one_row("2015-12-31"), _stores())
    future, _ = features.build(_one_row("2030-06-15"), _stores())

    gap = (
        future["competition_open_age_months"].iloc[0]
        - at_cap["competition_open_age_months"].iloc[0]
    )
    assert gap > 150


def test_the_cap_holds_the_age_at_its_end_of_data_value() -> None:
    at_cap, _ = features.build(_one_row("2015-12-31"), _stores(), age_as_of=CAP)
    future, _ = features.build(_one_row("2030-06-15"), _stores(), age_as_of=CAP)

    assert (
        future["competition_open_age_months"].iloc[0]
        == at_cap["competition_open_age_months"].iloc[0]
    )


def test_the_cap_leaves_the_calendar_alone() -> None:
    """Capping the calendar too would answer about the wrong weekday entirely."""
    built, _ = features.build(_one_row("2030-06-15"), _stores(), age_as_of=CAP)
    real = pd.Timestamp("2030-06-15")

    assert built["day_of_week"].iloc[0] == real.dayofweek
    assert built["month"].iloc[0] == real.month
    assert built["day_of_month"].iloc[0] == real.day


def test_a_date_inside_the_window_is_unaffected_by_the_cap() -> None:
    """The cap must be a no-op everywhere it is not needed, training included."""
    plain, _ = features.build(_one_row("2014-06-15"), _stores())
    capped, _ = features.build(_one_row("2014-06-15"), _stores(), age_as_of=CAP)

    assert (
        plain["competition_open_age_months"].iloc[0]
        == capped["competition_open_age_months"].iloc[0]
    )


# -- the guard, through the serving path ---------------------------------------


def test_the_artifact_carries_the_span_of_its_data(artifact) -> None:
    assert artifact.observed_from is not None
    assert artifact.observed_through is not None
    assert artifact.observed_from < artifact.observed_through


def test_serving_refuses_a_date_before_the_data(artifact) -> None:
    with pytest.raises(StockoutError, match="before the data"):
        serving_row(artifact, entity=artifact.entities[0], date="2009-01-01")


def test_serving_answers_a_date_after_the_data(artifact) -> None:
    """Previously refused at the HTTP boundary and unguarded everywhere else."""
    matrix = serving_row(artifact, entity=artifact.entities[0], date="2030-06-15")
    assert len(matrix) == 1
    assert list(matrix.columns) == list(artifact.feature_columns)


def test_a_prediction_past_the_data_is_marked_a_forecast(artifact) -> None:
    answer = predict(artifact, entity=artifact.entities[0], date="2030-06-15")
    assert answer.status is not None
    assert answer.status.is_forecast
    assert answer.status.caveat() is not None


def test_a_prediction_inside_the_data_is_marked_observed(artifact) -> None:
    answer = predict(artifact, entity=artifact.entities[0], date="2014-06-15")
    assert answer.status is not None
    assert not answer.status.is_forecast
    assert answer.status.caveat() is None


def test_the_cli_wording_says_so_on_a_forecast_and_not_otherwise(artifact) -> None:
    """`lines()` is what `stockout predict` prints, and it is the CLI's only chance to
    say that nothing measured backs the number."""
    inside = predict(artifact, entity=artifact.entities[0], date="2014-06-15")
    beyond = predict(artifact, entity=artifact.entities[0], date="2030-06-15")

    assert not any("forecast" in line for line in inside.lines())
    assert any("forecast" in line for line in beyond.lines())


def test_a_forecast_still_produces_a_usable_number(artifact) -> None:
    """Capping the ages must not collapse the answer to zero or a NaN."""
    answer = predict(artifact, entity=artifact.entities[0], date="2030-06-15")
    assert answer.sales > 0
    assert answer.demand_class in {"Low", "Medium", "High"}


def test_date_status_is_none_when_the_artifact_has_no_span(artifact) -> None:
    """A hand-built `Artifact` predates the field; answering without a claim is the only
    honest option, and `persistence.load` refuses such a file anyway."""
    import dataclasses

    stripped = dataclasses.replace(artifact, observed_from=None, observed_through=None)
    assert date_status(stripped, "2030-06-15") is None
