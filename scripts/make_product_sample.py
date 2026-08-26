"""Regenerate the committed store-item sample.

    python scripts/make_product_sample.py [n_stores] [n_items] [days]

Deterministic, so a run that produces a diff means the generator changed. One file, not
two: a product source has no per-entity metadata to join, which is the shape the real
Store Item Demand data ships and the reason the what-if page cannot serve it.
"""

from __future__ import annotations

import sys

from stockout.config import SAMPLE_PRODUCTS_PATH
from stockout.data.synth_products import (
    DEFAULT_DAYS,
    DEFAULT_ITEMS,
    DEFAULT_STORES,
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

    SAMPLE_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(SAMPLE_PRODUCTS_PATH, index=False)

    print(
        f"{SAMPLE_PRODUCTS_PATH.name}: {len(sales):,} rows x {len(sales.columns)} columns "
        f"| {n_stores} stores x {n_items} items x {days} days"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
