"""Answer one store-day: how much it will sell, and whether that is Low, Medium or High.

Both answers come from the artefact, and they come from two different models that can
disagree. Showing one while implying both would be the most misleading thing here, so
`Prediction` always carries the pair and the cut points that separate the classes.

The single serving row is built through exactly the same `features.build` the training
frame went through, with the encoders the artefact carries. That is the whole reason the
encoders travel: a second `LabelEncoder` fitted here would renumber the store types and
answer confidently with nonsense.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from . import features, targets, window
from .data import schemas as s
from .errors import StockoutError
from .persistence import Artifact
from .registry import model_names
from .window import DateStatus

#: A serving row has no sales yet. The column has to exist for the feature builder, and
#: nothing reads it — every feature is built from the calendar and the store metadata.
_SALES_PLACEHOLDER = 0.0

#: How many unknown entities an error names before it stops listing them.
_UNKNOWN_SHOWN = 3


@dataclass
class Prediction:
    """What the panel and the CLI both display."""

    entity: str
    date: pd.Timestamp
    sales: float
    demand_class: str
    thresholds: tuple[float, float]
    per_model: dict[str, float] = field(default_factory=dict)

    #: Whether a true figure for this store-day exists in the data behind the models.
    #: `None` where the artefact predates the field and the window is unknown. Carried on
    #: the answer rather than worked out by each caller, because the two regions mean
    #: different things and a surface that shows the same wording for both is misleading:
    #: inside the window the prediction can be checked against an actual, outside it the
    #: prediction is the only thing there is.
    status: DateStatus | None = None

    def threshold_text(self) -> str:
        low, high = self.thresholds
        return f"Low <{low:,.0f} | Medium {low:,.0f}-{high:,.0f} | High >{high:,.0f}"

    def lines(self) -> list[str]:
        out = [
            f"{self.entity} on {self.date:%Y-%m-%d}",
            f"  Predicted sales   {self.sales:,.0f}",
            f"  Demand class      {self.demand_class}",
            f"  cut points        {self.threshold_text()}",
        ]
        caveat = self.status.caveat() if self.status else None
        if caveat:
            out.append(f"  forecast          {caveat}")
        return out


def date_status(artifact: Artifact, date: str | pd.Timestamp) -> DateStatus | None:
    """Classify a request date against the artefact's window, or `None` if it has none.

    An artefact written before `observed_from` existed carries neither bound, and the only
    honest thing to do with it is what this package did before: answer without a claim
    either way. The version check in `persistence.load` refuses such a file outright, so
    in practice this returns `None` only for an `Artifact` built by hand in a test.
    """
    if artifact.observed_from is None or artifact.observed_through is None:
        return None
    return window.check(date, artifact.observed_from, artifact.observed_through)


def serving_matrix(
    artifact: Artifact,
    *,
    entities: Sequence[str],
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> pd.DataFrame:
    """The model matrix for many entities on one day, exactly as training built it.

    One frame and one `features.build` for the whole set. The ranked-items page scores
    every item a shop stocks against a single date, and doing that as a loop of one-row
    builds pays the pandas overhead once per item for an identical result — the work is in
    constructing frames, not in the arithmetic.

    The row order is the order asked for, so a caller can zip its own labels against the
    predictions without matching on the entity again.
    """
    names = [str(entity) for entity in entities]
    if not names:
        raise StockoutError("no entities to score")

    known = set(artifact.entities)
    if known:
        unknown = [name for name in names if name not in known]
        if unknown:
            shown = ", ".join(repr(name) for name in unknown[:_UNKNOWN_SHOWN])
            hidden = len(unknown) - _UNKNOWN_SHOWN
            more = "" if hidden <= 0 else f" and {hidden} more"
            raise StockoutError(
                f"unknown {artifact.entity_label} {shown}{more}; the model was fitted on "
                f"{len(artifact.entities):,} of them"
            )

    status = date_status(artifact, date)
    stamp = status.stamp if status else window.parse(date)

    rows = pd.DataFrame(
        [
            {
                s.ENTITY: name,
                s.DATE: stamp,
                s.SALES: _SALES_PLACEHOLDER,
                s.OPEN: 1,
                s.PROMO: int(promo),
                s.STATE_HOLIDAY: state_holiday,
                s.SCHOOL_HOLIDAY: int(school_holiday),
            }
            for name in names
        ]
    )
    built, _ = features.build(
        rows,
        artifact.stores,
        encoders=artifact.encoders,
        age_as_of=artifact.observed_through,
        entity_vocabulary=artifact.entity_vocabulary,
        has_products=artifact.has_products,
    )

    missing = [c for c in artifact.feature_columns if c not in built.columns]
    if missing:
        raise StockoutError(f"serving row is missing {', '.join(missing)}")
    return built[artifact.feature_columns]


def serving_row(
    artifact: Artifact,
    *,
    entity: str,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> pd.DataFrame:
    """Build the one-row model matrix for a store-day, exactly as training built it."""
    return serving_matrix(
        artifact,
        entities=[entity],
        date=date,
        promo=promo,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )


def predict(
    artifact: Artifact,
    *,
    entity: str,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
    all_models: bool = False,
) -> Prediction:
    """Both answers for one store-day, optionally with every regressor's own number."""
    matrix = serving_row(
        artifact,
        entity=entity,
        date=date,
        promo=promo,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )

    entity = str(entity)
    # The regressor answers in residuals when the artefact carries trends, so the store's
    # line goes back under it here. This is the whole of what lets a date past the data
    # differ from the same kind of day inside it — the tree returns the same leaf for
    # both, and only the line knows that the store is bigger by then.
    sales = _with_trend(artifact, float(artifact.regressor.predict(matrix)[0]), entity, date)
    demand = str(artifact.classifier.predict(matrix)[0])

    return Prediction(
        entity=entity,
        date=pd.to_datetime(date),
        sales=sales,
        demand_class=demand,
        thresholds=_thresholds_at(artifact, entity, date),
        per_model=_every_regressor(artifact, matrix) if all_models else {},
        status=date_status(artifact, date),
    )


def _thresholds_at(
    artifact: Artifact, entity: str, date: str | pd.Timestamp
) -> tuple[float, float]:
    """This store's demand cut points, in sales for the day being asked about.

    On a de-trended artefact the stored cut points are residuals — distances from the
    store's line — so they come out negative and mean nothing beside a sales figure. The
    panel printed `Low <-587` next to a prediction of 4,103, which is not a bug in the
    model but is certainly a lie to the reader.

    Shifting them onto the line at the requested date puts both back in the same units and
    restores what the pair is for: *this is the prediction, and here is where Low, Medium
    and High sit for this shop on this day*. The band therefore rises as the store grows,
    which is exactly the behaviour asked for when the class was defined relative to the
    store's current level rather than to an absolute figure fixed in 2015.
    """
    low, high = artifact.thresholds.for_entity(entity)
    if artifact.trends is None:
        return low, high
    level = artifact.trends.level_at(entity, pd.Timestamp(pd.to_datetime(date)))
    return low + level, high + level


def _with_trend(
    artifact: Artifact, residual: float, entity: str, date: str | pd.Timestamp
) -> float:
    """Add the store's fitted level back to a residual, if this artefact has lines."""
    if artifact.trends is None:
        return residual
    return residual + artifact.trends.level_at(entity, pd.Timestamp(pd.to_datetime(date)))


def _every_regressor(artifact: Artifact, matrix: pd.DataFrame) -> dict[str, float]:
    """What each registered regressor predicts for this row.

    Only the deployed model is refitted-and-ready; the others are not in the artefact, so
    this reports the deployed one and leaves the rest to the caller that has a training
    frame. The panel uses `compare_row` below instead.
    """
    return {artifact.regressor_name: float(artifact.regressor.predict(matrix)[0])}


def compare_row(
    artifact: Artifact,
    fitted: Mapping[str, object],
    *,
    entity: str,
    date: str | pd.Timestamp,
    promo: bool = False,
    state_holiday: str = "0",
    school_holiday: bool = False,
) -> dict[str, float]:
    """Every fitted regressor's prediction for one row, for the panel's comparison panel.

    `fitted` comes from the web service, which keeps the whole registry fitted in memory
    so the expandable section can show that the models disagree.

    Every field of the request is threaded, not just the promotion. Without the two
    holiday fields this answered an ordinary day while the headline above it answered the
    Christmas that was actually asked for, and the page presented the gap as two models
    disagreeing rather than as two different questions.
    """
    matrix = serving_row(
        artifact,
        entity=entity,
        date=date,
        promo=promo,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
    )
    answers: dict[str, float] = {}
    for name in model_names("regression"):
        model = fitted.get(name)
        if model is not None:
            # These are fitted through the same de-trended split as the deployed model,
            # so they answer in residuals too. Shown raw they would put a number near
            # zero beside a sales figure and look like a broken model rather than a
            # missing addition.
            residual = float(model.predict(matrix)[0])  # type: ignore[attr-defined]
            answers[name] = _with_trend(artifact, residual, str(entity), date)
    return answers


def classify(artifact: Artifact, sales: float, entity: str) -> str:
    """Turn a predicted number into a class with that entity's own cut points."""
    return targets.classify_one(sales, str(entity), artifact.thresholds)
