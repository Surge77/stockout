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

#: Columns that exist in training data but are unknown at the forecast origin.
#: `customers` is the whole reason this constant exists: it is the single best
#: predictor of `sales` and nobody knows it six weeks ahead. A model that uses it
#: scores brilliantly and cannot be deployed. See features/build.py.
UNAVAILABLE_AT_FORECAST_TIME: Final[frozenset[str]] = frozenset({CUSTOMERS})
