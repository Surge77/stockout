"""Regenerate the committed sample at data/sample_sales.csv.

Run after changing anything in `stockout.data.synth`:

    python scripts/make_sample.py

The generator is deterministic, so a run that produces a diff means the generator
changed. CI checks exactly that and warns rather than failing, because a deliberate
generator change should not block a merge — it should be noticed.
"""

from __future__ import annotations

from stockout.config import SAMPLE_PATH
from stockout.data.loaders import write_sales
from stockout.data.synth import make_sales
from stockout.data.validate import validate_sales

#: Four stores over two years: enough for five 42-day folds on a 365-day minimum
#: training window, and small enough (~100 KB) that committing it is not rude.
N_STORES = 4
DAYS = 730
SEED = 7


def main() -> int:
    frame = make_sales(n_stores=N_STORES, days=DAYS, seed=SEED)
    validate_sales(frame)
    path = write_sales(frame, SAMPLE_PATH)
    print(f"wrote {len(frame):,} rows across {N_STORES} stores to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
