"""Regenerate the committed store-item sample and its catalogue.

    python scripts/make_product_sample.py [n_stores] [n_items] [days]

Deterministic, so a run that produces a diff means the generator changed.

Two files, like the store-day sample: the sales panel, and one row per item saying what
that item is. The real Store Item Demand data ships only the first — it publishes an
integer per item and nothing else — so `store_item_demand` declares
`has_item_metadata=False` and this sample is where the product path's attributes come from.
"""

from __future__ import annotations

import sys

from stockout.config import SAMPLE_CATALOGUE_PATH, SAMPLE_PRODUCTS_PATH
from stockout.data.synth_products import (
    DEFAULT_DAYS,
    DEFAULT_ITEMS,
    DEFAULT_STORES,
    make_catalogue,
    make_product_sales,
)

START = "2013-01-01"
SEED = 7


def main(argv: list[str]) -> int:
    n_stores = int(argv[0]) if argv else DEFAULT_STORES
    n_items = int(argv[1]) if len(argv) > 1 else DEFAULT_ITEMS
    days = int(argv[2]) if len(argv) > 2 else DEFAULT_DAYS

    sales = make_product_sales(
        n_stores=n_stores, n_items=n_items, start=START, days=days, seed=SEED
    )

    catalogue = make_catalogue(n_items)

    SAMPLE_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(SAMPLE_PRODUCTS_PATH, index=False)
    catalogue.to_csv(SAMPLE_CATALOGUE_PATH, index=False)

    print(f"{SAMPLE_CATALOGUE_PATH.name}: {len(catalogue):,} items")
    print(
        f"{SAMPLE_PRODUCTS_PATH.name}: {len(sales):,} rows x {len(sales.columns)} columns "
        f"| {n_stores} stores x {n_items} items x {days} days"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
