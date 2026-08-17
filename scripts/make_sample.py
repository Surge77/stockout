"""Regenerate the committed samples at data/sample_sales.csv and data/sample_stores.csv.

Run after changing anything in `stockout.data.synth` or `stockout.data.synth_stores`:

    python scripts/make_sample.py

Both generators are deterministic, so a run that produces a diff means a generator
changed. CI checks exactly that and warns rather than failing, because a deliberate
generator change should not block a merge — it should be noticed.

Two files rather than one because the real data is two files. Flattening the join into
a single committed CSV would repeat each store's metadata across 730 rows and, worse,
would let the package be developed against a shape the downloaded data does not have.
"""

from __future__ import annotations

from stockout.config import SAMPLE_PATH, SAMPLE_STORES_PATH
from stockout.data.loaders import write_sales
from stockout.data.synth import make_sales
from stockout.data.synth_stores import make_stores
from stockout.data.validate import validate_sales

#: Four stores over two years: enough for five folds on a 365-day minimum training
#: window, and small enough (~100 KB) that committing it is not rude. Four is also the
#: number at which `synth_stores` guarantees every store type and every missing-value
#: pattern appears — see its module docstring.
N_STORES = 4
DAYS = 730
SEED = 7


def main() -> int:
    frame = make_sales(n_stores=N_STORES, days=DAYS, seed=SEED)
    validate_sales(frame)
    path = write_sales(frame, SAMPLE_PATH)
    print(f"wrote {len(frame):,} rows across {N_STORES} stores to {path}")

    stores = make_stores(n_stores=N_STORES, seed=SEED)
    stores_path = write_sales(stores, SAMPLE_STORES_PATH)
    print(f"wrote {len(stores)} store metadata rows to {stores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
