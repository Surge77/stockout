"""Build the model matrix from things that are knowable before the day happens.

**No lags, no rolling means.** Every feature here comes from the calendar, from the
store's own metadata, or from a flag that is decided in advance — a promotion calendar for
next month is already sitting in a planning system. Nothing is derived by shifting the
sales column, so there is nothing for a shuffled split to leak through and
`train_test_split` stays defensible.

What that costs, stated rather than hidden: this answers *what sales should we expect at
this store on this kind of day*, not *forecast the next six weeks*. Two adjacent days at
one store are near-duplicates, so a shuffled split still flatters the score a little.
Much smaller than a lag leak, not zero.

The matrix is assembled from three families, each in its own module where it is big enough
to warrant one: the calendar and the product axes below, the store metadata in
`store_features.py`, and every id-to-integer mapping in `encoding.py`. `build` is the only
way in, so the training path and the serving path cannot drift — the web panel scores a
single row through exactly the code that built the rows the model was fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data import schemas as s
from .encoding import (
    LABEL_VOCABULARIES,
    UNKNOWN,
    UNKNOWN_ENTITY_CODE,
    as_code,
    encode_labels,
    entity_codes,
)
from .store_features import MONTH_TOKENS, STORE_FEATURES, add_store_features

#: Re-exported so callers keep importing feature vocabulary from one place. The split
#: below is about file size, not about asking every caller to learn a new import path.
__all__ = [
    "CALENDAR_FEATURES",
    "ENTITY_CODE",
    "ITEM_FEATURES",
    "LABEL_VOCABULARIES",
    "MONTH_TOKENS",
    "PRODUCT_FEATURES",
    "ROW_FEATURES",
    "STORE_FEATURES",
    "UNKNOWN",
    "UNKNOWN_ENTITY_CODE",
    "add_calendar_features",
    "add_product_features",
    "add_store_features",
    "build",
    "drop_closed",
    "encode_labels",
    "feature_columns",
    "is_constant",
]

CALENDAR_FEATURES: tuple[str, ...] = (
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
)

ROW_FEATURES: tuple[str, ...] = (s.PROMO, s.STATE_HOLIDAY, s.SCHOOL_HOLIDAY)

#: The two axes of a store-item entity, as separate features. Built only for sources that
#: have a product axis, so a store-day model's matrix is unchanged by their existence.
#:
#: Separate, and that is the whole design. `entity_code` identifies the series — store 3's
#: item 25 — and a model given only that has to learn every one of the 200 levels on its
#: own rows. Given the two axes it can learn twenty item levels from ten stores' worth of
#: rows each and ten store multipliers from twenty items' worth, then compose them. Fewer
#: quantities, each estimated from more data.
#:
#: Measured on `synthetic_products`, gradient boosting, five held-out pairs: the axes are
#: worth about **+0.009 r²** on an ordinary held-out split (0.8965 -> 0.9058), which is
#: small — every series has hundreds of its own rows there, so identity alone is nearly
#: enough and this is the wrong test for them.
#:
#: What they buy shows up on a store-item pair held out **entirely**: mean r² **-14.39**
#: from identity alone against **-0.39** from the two axes. A code the fit never saw lands
#: in whichever leaf its integer happens to fall down; a store and an item the fit has both
#: seen thousands of times still compose. Both are negative — neither beats answering with
#: that pair's own mean — but they are two different orders of failure, and only one of
#: them is usable as a starting estimate for a line a shop has not stocked before.
#:
#: `entity_code` is kept alongside them rather than replaced. The pair are additive by
#: construction; anything specific to one store's copy of one item — a local favourite —
#: lives in the identity and nowhere else.
PRODUCT_FEATURES: tuple[str, ...] = ("store_code", "item_code")

#: What an item *is*, where the source says so. Separate from `PRODUCT_FEATURES` because
#: the two answer different questions: those are identity, this is description.
#:
#: The distinction is what makes a described item answerable at all. `item_code` cannot
#: help with a product the fit has never seen — there is no id to supply — whereas a
#: category it has seen thousands of rows of still carries a level. That is the same split
#: `store_type`/`assortment` make for shops, and the reason the described-item page exists
#: on a source with a catalogue and refuses on one without.
ITEM_FEATURES: tuple[str, ...] = (s.CATEGORY,)

#: The entity's integer id. Never one-hot: real Rossmann has 1,115 stores, which would be
#: 1,115 columns nobody can defend and a distance metric where every pair of stores is
#: equally far apart.
ENTITY_CODE = "entity_code"

_SATURDAY = 5


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Everything derivable from the date alone. Knowable arbitrarily far ahead."""
    frame = frame.copy()
    dates = pd.to_datetime(frame[s.DATE])
    frame["day_of_week"] = dates.dt.dayofweek.astype("int16")
    frame["month"] = dates.dt.month.astype("int16")
    frame["day_of_month"] = dates.dt.day.astype("int16")
    frame["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    frame["is_weekend"] = (dates.dt.dayofweek >= _SATURDAY).astype("int8")
    return frame


def add_product_features(frame: pd.DataFrame, *, has_products: bool) -> pd.DataFrame:
    """Split a store-item entity into its two axes, from whichever shape is to hand.

    Training frames arrive from the adapter with `store` and `item` still on them, so the
    columns are read directly. A serving request carries an entity and nothing else — the
    web form knows it picked store 3 and item 25, but `serving_row` builds the same
    canonical row every source uses — so the pair is recovered by splitting the id.

    Both routes must agree, because one fits the model and the other queries it. Reading
    the columns where they exist and splitting where they do not is what keeps them
    agreeing without the serving path having to reconstruct an adapter's output.

    `has_products` is passed rather than sniffed. Deciding by "does the entity contain a
    separator" would make a store-day source's behaviour depend on how its ids happen to
    be spelled, which is exactly the kind of implicit contract that breaks quietly when a
    new source arrives.
    """
    if not has_products:
        return frame

    frame = frame.copy()
    if s.STORE in frame.columns and s.ITEM in frame.columns:
        store, item = frame[s.STORE], frame[s.ITEM]
    else:
        halves = frame[s.ENTITY].astype("string").str.split(
            s.PRODUCT_ENTITY_SEPARATOR, n=1, expand=True
        )
        store, item = halves[0], halves[1]

    frame["store_code"] = as_code(store)
    frame["item_code"] = as_code(item)
    return frame


def feature_columns(
    frame: pd.DataFrame,
    *,
    has_promotions: bool = True,
    has_store_metadata: bool = True,
    has_item_metadata: bool = False,
) -> list[str]:
    """The model matrix's columns, in a stable order.

    A whitelist, which is what keeps both targets out of the matrix: `sales` and
    `demand_class` are excluded by not appearing here, rather than by a blocklist that has
    to be kept in step with every column any adapter might add.

    The two flags drop whole families the **source declares it does not carry**. A product
    source records no promotion and no store attributes, so the adapter supplies those
    canonical columns as constants to keep one shape across every source — and a constant
    column is not a feature. Ten of a product matrix's eighteen columns were filler that
    no model can split on, cluttering `describe`, the importances and this list.

    Declared absence, deliberately, rather than measured constancy. `synthetic` happens to
    hold one value in `state_holiday` too, and dropping columns because *this sample* never
    varied would silently change the matrix a published score was measured on.
    """
    wanted = [
        ENTITY_CODE,
        *PRODUCT_FEATURES,
        *(ITEM_FEATURES if has_item_metadata else ()),
        *CALENDAR_FEATURES,
        *(ROW_FEATURES if has_promotions else ()),
        *(STORE_FEATURES if has_store_metadata else ()),
    ]
    return [column for column in wanted if column in frame.columns]


def build(
    sales: pd.DataFrame,
    stores: pd.DataFrame | None,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    age_as_of: pd.Timestamp | None = None,
    entity_vocabulary: list[str] | None = None,
    has_products: bool = False,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Sales rows plus optional metadata in, model-ready frame out.

    `age_as_of` is a serving concern only — see `store_features.add_store_features`.
    Training leaves it unset, so the frame the models are fitted on is unchanged by its existence.

    `entity_vocabulary` is the other serving concern — see `encoding.entity_codes`. Training
    derives it from the frame and the artefact carries it, because a one-row request
    cannot recover the ordering a million-row fit produced.

    `has_products` says whether `entity` is a store-item pair, and comes from the source
    rather than from the shape of the ids — see `add_product_features`.
    """
    frame = add_calendar_features(sales)
    frame = add_store_features(frame, stores, age_as_of=age_as_of)
    frame = add_product_features(frame, has_products=has_products)
    frame[ENTITY_CODE] = entity_codes(frame[s.ENTITY], entity_vocabulary)
    frame, encoders = encode_labels(frame, encoders=encoders)

    for column in feature_columns(frame):
        if column not in LABEL_VOCABULARIES:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame, encoders


def drop_closed(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove days the shop was shut.

    A closed store sells exactly zero and every model predicts zero, so keeping those
    rows adds a large block of free perfect predictions — on Rossmann roughly a seventh
    of them. Every score in this package would be flattered by an amount bought entirely
    by predicting that a shut shop sells nothing.
    """
    if s.OPEN not in frame.columns:
        return frame
    return frame[frame[s.OPEN] == 1].reset_index(drop=True)


def is_constant(values: pd.Series) -> bool:
    """True when a column carries no information, so callers can skip it."""
    return bool(np.asarray(values.nunique(dropna=False)) <= 1)
