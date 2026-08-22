"""Regenerate the committed retail sample.

    python scripts/make_sample.py [n_stores] [days]

Both generators are deterministic, so a run that produces a diff means a generator
changed. Two files rather than one because the real Rossmann data is two files, and
flattening the join would repeat each store's metadata across every one of its rows.
"""

from __future__ import annotations

import sys

from stockout.config import SAMPLE_PATH, SAMPLE_STORES_PATH
from stockout.data.synth import make_sales
from stockout.data.synth_stores import make_stores

DEFAULT_STORES = 50
DEFAULT_DAYS = 1095
START = "2013-01-01"
SEED = 7


def main(argv: list[str]) -> int:
    n_stores = int(argv[0]) if argv else DEFAULT_STORES
    days = int(argv[1]) if len(argv) > 1 else DEFAULT_DAYS

    sales = make_sales(n_stores=n_stores, start=START, days=days, seed=SEED)
    stores = make_stores(n_stores=n_stores, seed=SEED)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(SAMPLE_PATH, index=False)
    stores.to_csv(SAMPLE_STORES_PATH, index=False)

    print(f"{SAMPLE_PATH.name}: {len(sales):,} rows x {len(sales.columns)} columns")
    print(f"{SAMPLE_STORES_PATH.name}: {len(stores):,} rows x {len(stores.columns)} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
