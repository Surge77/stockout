"""Every assumption about the data's shape, in one place.

Column names are canonical snake_case internally. Rossmann's CSV ships PascalCase,
so `ROSSMANN_RENAME` is the only place that spelling exists — a schema change on
Kaggle's side breaks one dict, not thirty call sites.
"""

from __future__ import annotations

from typing import Final

DATE: Final = "date"
STORE: Final = "store"
SALES: Final = "sales"
CUSTOMERS: Final = "customers"
OPEN: Final = "open"
PROMO: Final = "promo"
STATE_HOLIDAY: Final = "state_holiday"
SCHOOL_HOLIDAY: Final = "school_holiday"
DAY_OF_WEEK: Final = "day_of_week"

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

#: The keys that identify one observation. Duplicates here are a data bug, not a tie.
KEY_COLUMNS: Final[tuple[str, ...]] = (STORE, DATE)

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
#: competitor distance means. `features/preprocess.py` decides it in the open.
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

# ---------------------------------------------------------------------------
# Feature roles. `features/preprocess.py` builds one ColumnTransformer branch per
# role, so these four tuples are the single place that decides how a column is
# encoded. Adding a column to the data without adding it here means it is not a
# feature — which is the safe default.
# ---------------------------------------------------------------------------

#: One-hot encoded. Every one is low-cardinality: 4, 3, 4 and 7 levels.
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    STORE_TYPE,
    ASSORTMENT,
    STATE_HOLIDAY,
    DAY_OF_WEEK,
)

#: Ordinal-encoded, deliberately. `store` has 1115 levels; one-hot encoding it
#: makes the design matrix sparse, which HistGradientBoosting refuses outright and
#: which StandardScaler would densify into several gigabytes. ADR 0019.
ORDINAL_FEATURES: Final[tuple[str, ...]] = (STORE,)

#: Tf-idf vectorised. `promo_interval` is a comma-separated month list such as
#: "Jan,Apr,Jul,Oct" — genuine text, and the only text this dataset has.
TEXT_FEATURE: Final = PROMO_INTERVAL

#: Imputed then scaled. Everything numeric that is not a key, a target or a role
#: above; `features/build.py` appends the generated calendar and lag columns.
STORE_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    COMPETITION_DISTANCE,
    COMPETITION_OPEN_MONTH,
    COMPETITION_OPEN_YEAR,
    PROMO2,
    PROMO2_SINCE_WEEK,
    PROMO2_SINCE_YEAR,
)

#: Columns that exist in training data but are unknown at the forecast origin.
#: `customers` is the whole reason this constant exists: it is the single best
#: predictor of `sales` and nobody knows it six weeks ahead. A model that uses it
#: scores brilliantly and cannot be deployed. See features/build.py.
UNAVAILABLE_AT_FORECAST_TIME: Final[frozenset[str]] = frozenset({CUSTOMERS})
