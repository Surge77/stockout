"""Load a retail source, build its features, and hand back a model-ready frame.

This is the seam between `data/adapters.py`, which knows what each source's columns are
called, and everything downstream, which only ever sees the canonical shape.

`prepare` is deliberately the only way in. Building features in one place means the
training path and the serving path cannot drift: the web panel scores a single row through
exactly the code that built the 54,000 rows the model was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import features
from .config import Task
from .data import schemas as s
from .data.adapters import SOURCE_NAMES, Source, source_for
from .errors import StockoutError

DATASET_NAMES: tuple[str, ...] = SOURCE_NAMES


@dataclass
class Prepared:
    """A model-ready frame plus everything needed to reproduce it at serving time."""

    frame: pd.DataFrame
    feature_columns: list[str]
    encoders: dict[str, LabelEncoder]
    source: Source
    stores: pd.DataFrame | None

    #: The entity ordering `features._entity_codes` was built against, or `None` where the
    #: ids are numeric and are their own code. Travels into the artefact because a one-row
    #: serving frame cannot recover the ordering a whole-frame fit produced — see
    #: `features._entity_codes` for what went wrong before it did.
    entity_vocabulary: list[str] | None = None

    @property
    def name(self) -> str:
        return self.source.name

    def entities(self) -> list[str]:
        return sorted(self.frame[s.ENTITY].astype("string").unique().tolist(), key=_sortable)

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        dates = pd.to_datetime(self.frame[s.DATE])
        return dates.min(), dates.max()


def _sortable(entity: str) -> tuple[int, float, str]:
    """Sort store "9" before "10" where the ids are numeric, alphabetically otherwise."""
    try:
        return (0, float(entity), "")
    except ValueError:
        return (1, 0.0, entity)


def dataset_for(name: str) -> Source:
    return source_for(name)


def prepare(name: str, *, drop_closed: bool = True) -> Prepared:
    """Load one source and build the model matrix.

    Closed days are dropped by default: a shut store sells exactly zero and every model
    predicts zero, so keeping them hands every score a large block of free correct
    answers — on Rossmann about a seventh of all rows.
    """
    source = source_for(name)
    sales, stores = source.load()
    if drop_closed:
        sales = features.drop_closed(sales)
    if sales.empty:
        raise StockoutError(f"{name} has no open days to model")

    vocabulary = _vocabulary_for(sales[s.ENTITY])
    frame, encoders = features.build(
        sales,
        stores,
        entity_vocabulary=vocabulary,
        has_products=source.has_products,
    )
    return Prepared(
        frame=frame,
        feature_columns=features.feature_columns(
            frame,
            has_promotions=source.has_promotions,
            has_store_metadata=source.has_store_metadata,
            has_item_metadata=source.has_item_metadata,
        ),
        encoders=encoders,
        source=source,
        stores=stores,
        entity_vocabulary=vocabulary,
    )


def _vocabulary_for(entities: pd.Series) -> list[str] | None:
    """The entity ordering serving will need, or `None` where the ids are their own code.

    Numeric ids — Rossmann's store numbers — are returned as `None` deliberately. Routing
    them through a vocabulary would replace store 1115's code with its position in the
    list, renumbering every store and moving scores that are already published, to fix a
    problem that path does not have.
    """
    names = entities.astype("string")
    if not pd.to_numeric(names, errors="coerce").isna().any():
        return None
    return sorted(names.unique().tolist(), key=_sortable)


def xy(prepared: Prepared, task: Task) -> tuple[pd.DataFrame, pd.Series]:
    """The feature matrix and the target for one task.

    Regression predicts `sales`. Classification predicts `demand_class`, which
    `targets.attach` must have added first — it depends on cut points fitted on the
    training rows alone, so it cannot be built here without leaking.
    """
    matrix = prepared.frame[prepared.feature_columns]
    if task == "regression":
        return matrix, prepared.frame[s.SALES]
    if s.DEMAND_CLASS not in prepared.frame.columns:
        raise StockoutError(
            "no demand_class column; fit thresholds on the training rows and attach it first"
        )
    return matrix, prepared.frame[s.DEMAND_CLASS]


def describe(prepared: Prepared) -> pd.DataFrame:
    """Per column: dtype, nulls, and the five-number summary."""
    frame = prepared.frame
    numeric = frame.select_dtypes("number")
    summary = numeric.describe().transpose()
    summary.insert(0, "dtype", frame[numeric.columns].dtypes.astype(str))
    summary.insert(1, "nulls", frame[numeric.columns].isnull().sum())
    summary.insert(2, "role", [_role(c, prepared) for c in numeric.columns])
    return summary


def _role(column: str, prepared: Prepared) -> str:
    if column == s.SALES:
        return "target (regression)"
    if column == s.DEMAND_CLASS:
        return "target (classification)"
    if column in prepared.feature_columns:
        return "feature"
    return "unused"


def summary_line(prepared: Prepared) -> str:
    start, end = prepared.date_range()
    return (
        f"{prepared.name}: {len(prepared.frame):,} open {prepared.source.entity_label}-days | "
        f"{len(prepared.entities()):,} {prepared.source.entity_label}s | "
        f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    )
