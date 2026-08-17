"""Answering "what will store 3 sell next Tuesday", which is harder than it sounds.

The course notebooks end with a cell that builds a one-row DataFrame and calls
`model.predict`. That works when every feature is something the user types. Here most of
them are not: `sales_lag_7`, `sales_roll_mean_28` and the rest are computed *from the
store's own history*, and a row invented in isolation has none.

So a prediction here is three steps, not one:

1. Take the history the model was trained against.
2. Append the day being asked about, carrying its future-known covariates — the
   promotion calendar, the holiday flags, whether the shop is open. Those come from a
   planning system and genuinely are known in advance.
3. Rebuild the features over the combined frame and read off the last row.

Step 3 calls the same `build_features` the training path calls. Recomputing the lags with
a second, hand-written implementation is how a served prediction quietly stops matching
the model that was validated.

**There is a horizon-shaped limit on how far ahead this can see, and it is not a bug.**
The model reads `sales_lag_{horizon}`. To predict day D it needs the actual sales of day
`D - horizon`. If history ends on day H then D can be at most `H + horizon`; past that,
the lag would have to be a prediction of a prediction, and each one would inherit the
last one's error. Asking beyond the limit raises rather than quietly compounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DEMAND_CLASS_LABELS
from .data import schemas as s
from .errors import BacktestError, SchemaError
from .features.build import build_features
from .features.calendar import CALENDAR_COLUMNS
from .features.store_features import STORE_FEATURE_COLUMNS, add_store_features
from .persistence import Artifact
from .targets import DEMAND_CLASS, DEMAND_CLASS_CODE

#: Columns `build_features` generates, matched by prefix rather than by name because the
#: exact set depends on the horizon and the rolling windows in force.
_GENERATED_PREFIXES: tuple[str, ...] = (f"{s.SALES}_lag_", f"{s.SALES}_roll_")


@dataclass(frozen=True)
class Forecast:
    """One store, one day, both halves of the answer."""

    store: int
    date: pd.Timestamp
    sales: float
    demand_class: str
    thresholds: tuple[float, float]

    def summary(self) -> str:
        low, high = self.thresholds
        return (
            f"store {self.store} on {self.date.date()}: "
            f"{self.sales:,.0f} predicted, class {self.demand_class} "
            f"(this store's cut points are {low:,.0f} and {high:,.0f})"
        )


def forecast(
    artifact: Artifact,
    history: pd.DataFrame,
    *,
    store: int,
    date: str | pd.Timestamp,
    promo: int = 0,
    school_holiday: int = 0,
    state_holiday: str = "0",
    is_open: int = 1,
) -> Forecast:
    """Predict one store-day, using `history` to supply the lags.

    `history` is a prepared frame — whatever `dataset.prepare` produced for training. The
    generated columns are stripped and rebuilt rather than trusted, because the row being
    added changes the rolling windows that end on it.
    """
    when = pd.Timestamp(date)
    source = _source_rows(history, store=store)
    _require_answerable(source, when=when, horizon=artifact.horizon, store=store)

    future = _future_row(
        source,
        store=store,
        when=when,
        promo=promo,
        school_holiday=school_holiday,
        state_holiday=state_holiday,
        is_open=is_open,
    )
    combined = pd.concat([source, future], ignore_index=True)
    featured = build_features(add_store_features(combined), horizon=artifact.horizon)
    target = featured.iloc[[-1]]

    sales = float(artifact.regressor.predict(target).iloc[0])
    cuts = artifact.thresholds.for_store(store)
    return Forecast(
        store=store,
        date=when,
        sales=sales,
        demand_class=_label(artifact, target, sales=sales, cuts=cuts),
        thresholds=cuts,
    )


def _label(
    artifact: Artifact, target: pd.DataFrame, *, sales: float, cuts: tuple[float, float]
) -> str:
    """The classifier's answer, falling back to binning the regression when it abstains.

    The classifier returns `pd.NA` for a closed day, which is correct — a shut store has
    no demand class. Rather than surface a null into a web page, the fallback bins the
    predicted number against the same cut points, which is the "regress then bin" answer
    to the question every examiner asks about why both models exist.
    """
    predicted = artifact.classifier.predict(target)
    code = predicted.iloc[0]
    if pd.isna(code):
        low, high = cuts
        code = int(sales > low) + int(sales > high)
    return DEMAND_CLASS_LABELS[int(code)]


def _source_rows(history: pd.DataFrame, *, store: int) -> pd.DataFrame:
    """This store's own history, with the generated columns removed.

    Per store, because the lags are built with `groupby(store).shift()` and the other
    1,114 stores contribute nothing to them. On Rossmann that is the difference between
    rebuilding features over a thousand rows and over a million, per request.
    """
    if s.STORE not in history.columns or s.DATE not in history.columns:
        raise SchemaError("history needs store and date columns")

    rows = history[history[s.STORE] == store]
    if rows.empty:
        raise BacktestError(f"no history for store {store}")
    return rows[_source_columns(rows)].sort_values(s.DATE).reset_index(drop=True)


def _source_columns(frame: pd.DataFrame) -> list[str]:
    """Everything the feature builders did not create.

    Computed by subtraction rather than by listing what to keep, so a column added to the
    raw data upstream reaches the rebuild without anybody remembering to whitelist it.
    """
    generated = {*CALENDAR_COLUMNS, *STORE_FEATURE_COLUMNS, DEMAND_CLASS, DEMAND_CLASS_CODE}
    return [
        column
        for column in frame.columns
        if column not in generated and not column.startswith(_GENERATED_PREFIXES)
    ]


def _require_answerable(
    source: pd.DataFrame, *, when: pd.Timestamp, horizon: int, store: int
) -> None:
    last = pd.Timestamp(source[s.DATE].max())
    limit = last + pd.Timedelta(days=horizon)

    if when <= last:
        raise BacktestError(
            f"{when.date()} is already in store {store}'s history, which ends "
            f"{last.date()}. This forecasts the future; use the backtest to score the past."
        )
    if when > limit:
        raise BacktestError(
            f"cannot forecast {when.date()}: store {store}'s history ends {last.date()} "
            f"and the model reads sales from {horizon} days before the day it predicts, "
            f"so it can see to {limit.date()} at most. Predicting further would feed the "
            "model its own output as an observation."
        )


def _future_row(
    source: pd.DataFrame,
    *,
    store: int,
    when: pd.Timestamp,
    promo: int,
    school_holiday: int,
    state_holiday: str,
    is_open: int,
) -> pd.DataFrame:
    """One row for the day being asked about.

    The store's metadata is copied from its most recent row: `store_type`,
    `competition_distance` and the rest describe the shop rather than the day, and a
    caller should not have to restate them to ask a question.

    `sales` is NaN, not zero. It is the thing being predicted, and a zero would be
    indistinguishable from a closed day to every rolling window that ends on this row.
    """
    row: dict[str, object] = {
        str(key): value for key, value in source.iloc[-1].to_dict().items()
    }

    row[s.DATE] = when
    row[s.STORE] = store
    row[s.DAY_OF_WEEK] = when.dayofweek + 1  # Rossmann counts Monday as 1, pandas as 0
    row[s.SALES] = float("nan")
    row[s.OPEN] = is_open
    row[s.PROMO] = promo
    row[s.SCHOOL_HOLIDAY] = school_holiday
    row[s.STATE_HOLIDAY] = state_holiday
    if s.CUSTOMERS in row:
        row[s.CUSTOMERS] = float("nan")

    return pd.DataFrame([row]).astype(source.dtypes.to_dict())
