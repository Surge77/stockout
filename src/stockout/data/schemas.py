"""Every assumption about the data's shape, in one place.

Column names are canonical snake_case internally. Rossmann's CSV ships PascalCase,
so `ROSSMANN_RENAME` is the only place that spelling exists — a schema change on
Kaggle's side breaks one dict, not thirty call sites.
"""

from __future__ import annotations

from typing import Final

DATE: Final = "date"
STORE: Final = "store"

#: What joins a store id and an item id into one `entity`, and splits them back apart at
#: serving time. A contract between `adapters._as_product_panel`, which builds the string,
#: and `features.add_product_features`, which reverses it from a request that carries an
#: entity and nothing else. Store ids and item ids are integers on every product source, so
#: the separator cannot occur inside either half.
PRODUCT_ENTITY_SEPARATOR: Final = "_"

#: What an item *is*, on the one source that says. The real Store Item Demand file
#: publishes an integer and nothing else, so these exist for `synthetic_products` and
#: `Source.has_item_metadata` is what declares which sources carry them.
PRODUCT_NAME: Final = "product_name"
CATEGORY: Final = "category"

#: The product axis, on sources that have one. Absent from every store-day source, which
#: is why `features.PRODUCT_FEATURES` is built conditionally rather than always.
ITEM: Final = "item"
SALES: Final = "sales"
#: Recorded in the training data and unknowable at the forecast origin: it is the single
#: best predictor of `sales` and nobody knows it six weeks ahead. Never a feature, which
#: `features.feature_columns` secures by omission — see the note on `DEMAND_CLASS`.
CUSTOMERS: Final = "customers"
OPEN: Final = "open"
PROMO: Final = "promo"
STATE_HOLIDAY: Final = "state_holiday"
SCHOOL_HOLIDAY: Final = "school_holiday"
DAY_OF_WEEK: Final = "day_of_week"

#: The classification target: a tercile of `sales`, and therefore a deterministic function
#: of the regression target. Neither may ever be a feature of the other, and nothing here
#: enforces that — `features.feature_columns` is a whitelist, so a target is excluded by
#: not appearing on it rather than by being named on a blocklist that has to be maintained.
DEMAND_CLASS: Final = "demand_class"

# ---------------------------------------------------------------------------
# store.csv — one row per store, joined on `store`. These are the columns that
# make the preprocessing pipeline necessary rather than decorative: two are
# categorical, five are numeric with genuine missing values, and one is text.
# ---------------------------------------------------------------------------
STORE_TYPE: Final = "store_type"
ASSORTMENT: Final = "assortment"
COMPETITION_DISTANCE: Final = "competition_distance"
COMPETITION_OPEN_MONTH: Final = "competition_open_since_month"
COMPETITION_OPEN_YEAR: Final = "competition_open_since_year"
PROMO2: Final = "promo2"
PROMO2_SINCE_WEEK: Final = "promo2_since_week"
PROMO2_SINCE_YEAR: Final = "promo2_since_year"
PROMO_INTERVAL: Final = "promo_interval"

#: What demand is predicted *for*. Every adapter emits it, which is what lets one
#: registry, one comparison table and one web panel serve every source rather than one
#: each. `adapters.SOURCES` is the list of what it can currently hold.
ENTITY: Final = "entity"

#: The columns every adapter must emit. Anything beyond these is a dataset extra.
CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    ENTITY,
    DATE,
    SALES,
    OPEN,
    PROMO,
    STATE_HOLIDAY,
    SCHOOL_HOLIDAY,
)

#: The keys that identify one observation. Duplicates here are a data bug, not a tie.
KEY_COLUMNS: Final[tuple[str, ...]] = (STORE, DATE)

#: What identifies one observation on a source with a product axis. A store-day source
#: uses `KEY_COLUMNS`; on a product source the same pair would report every item of a
#: store-day as a duplicate of the others.
PRODUCT_KEY_COLUMNS: Final[tuple[str, ...]] = (STORE, ITEM, DATE)

#: What must be present before anything downstream will run.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    DATE,
    STORE,
    SALES,
    OPEN,
    PROMO,
    STATE_HOLIDAY,
    SCHOOL_HOLIDAY,
    DAY_OF_WEEK,
)

#: Rossmann DayOfWeek is 1=Monday .. 7=Sunday, which is *not* pandas' 0..6. Kept in
#: the CSV's convention so a value can be compared against the source without a
#: mental offset; `SUNDAY` names the one that matters.
SUNDAY: Final = 7

DTYPES: Final[dict[str, str]] = {
    STORE: "int32",
    SALES: "float64",
    CUSTOMERS: "float64",
    OPEN: "int8",
    PROMO: "int8",
    SCHOOL_HOLIDAY: "int8",
    STATE_HOLIDAY: "string",
    DAY_OF_WEEK: "int8",
}

ROSSMANN_RENAME: Final[dict[str, str]] = {
    "Date": DATE,
    "Store": STORE,
    "Sales": SALES,
    "Customers": CUSTOMERS,
    "Open": OPEN,
    "Promo": PROMO,
    "StateHoliday": STATE_HOLIDAY,
    "SchoolHoliday": SCHOOL_HOLIDAY,
    "DayOfWeek": DAY_OF_WEEK,
}

#: Kaggle's Store Item Demand data ships four lower-case columns, three of which already
#: match. `item` is the one this package had no name for until now.
STORE_ITEM_RENAME: Final[dict[str, str]] = {
    "date": DATE,
    "store": STORE,
    "item": ITEM,
    "sales": SALES,
}

ROSSMANN_STORE_RENAME: Final[dict[str, str]] = {
    "Store": STORE,
    "StoreType": STORE_TYPE,
    "Assortment": ASSORTMENT,
    "CompetitionDistance": COMPETITION_DISTANCE,
    "CompetitionOpenSinceMonth": COMPETITION_OPEN_MONTH,
    "CompetitionOpenSinceYear": COMPETITION_OPEN_YEAR,
    "Promo2": PROMO2,
    "Promo2SinceWeek": PROMO2_SINCE_WEEK,
    "Promo2SinceYear": PROMO2_SINCE_YEAR,
    "PromoInterval": PROMO_INTERVAL,
}

#: Left as float64 even where the values are conceptually integers: every one of
#: these is missing for some stores, and an int column cannot hold a NaN. Imputing
#: at read time would decide, silently and in the wrong place, what a missing
#: competitor distance means. `features.py` decides it in the open.
STORE_DTYPES: Final[dict[str, str]] = {
    STORE: "int32",
    STORE_TYPE: "string",
    ASSORTMENT: "string",
    COMPETITION_DISTANCE: "float64",
    COMPETITION_OPEN_MONTH: "float64",
    COMPETITION_OPEN_YEAR: "float64",
    PROMO2: "int8",
    PROMO2_SINCE_WEEK: "float64",
    PROMO2_SINCE_YEAR: "float64",
    PROMO_INTERVAL: "string",
}
