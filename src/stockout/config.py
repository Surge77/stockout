"""Paths and the two split constants every command shares."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = _ROOT / "data"

#: The committed retail sample: store-day sales, and one row per store of metadata.
#: Two files rather than one because the real Rossmann data is two files, and
#: flattening the join would repeat each store's metadata across every one of its rows.
SAMPLE_PATH = DATA_DIR / "sample_sales.csv"
SAMPLE_STORES_PATH = DATA_DIR / "sample_stores.csv"

#: The committed product sample: one row per store-item-day, in the same four columns the
#: real Store Item Demand file ships. One file rather than two because a product source
#: has no per-entity metadata to join — see `data/synth_products.py`.
SAMPLE_PRODUCTS_PATH = DATA_DIR / "sample_product_sales.csv"

#: Downloaded sources land here. Never committed: both Rossmann and Store Item Demand
#: forbid redistribution, which is why the two synthetic samples exist at all.
RAW_DIR = DATA_DIR / "raw"

#: Trained models land here. `persistence.load` refuses to read from anywhere else,
#: because loading a joblib file executes code from it.
ARTIFACT_DIR = _ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

#: Passed to every estimator that draws random numbers and to `train_test_split`, so
#: two runs of the same command produce the same table.
RANDOM_SEED = 42

#: The held-out fraction. 0.2 throughout, as taught.
TEST_SIZE = 0.2

#: How the held-out rows are chosen.
#:
#: `"shuffled"` is `train_test_split`, which is what the course teaches and what this
#: package used everywhere until it was measured. `"chronological"` holds out the last
#: `TEST_SIZE` of the calendar instead.
SplitStrategy = Literal["chronological", "shuffled"]

#: Chronological by default, and the reason is not style.
#:
#: Nothing in `features.py` is a lag, so the textbook leak — a model shown yesterday's
#: sales when predicting today — cannot happen here. A shuffled split still flatters the
#: score. `entity_code` together with `month`, `day_of_month` and `day_of_week` nearly
#: identifies one row, so a tree can put a test row in a leaf grown from the days on
#: either side of it, and the synthetic residual is an AR(1) with correlation 0.7 at one
#: day. The neighbours are informative, the split hands them over, and the number goes up
#: without the model having got better.
#:
#: `evaluate.split(..., strategy="shuffled")` still does the taught thing, and
#: `leakage.measure` reports the gap between the two.
DEFAULT_SPLIT: SplitStrategy = "chronological"

#: Whether to model each store's level separately from its seasonality.
#:
#: On by default, and this one is worth more than every other setting here combined.
#: Tree ensembles cannot extrapolate: past the largest threshold they ever split on, they
#: return the nearest leaf and stop. Rossmann grows about 5% a year, so a chronological
#: hold-out puts every test row in a period whose level the tree has no way to reach.
#: The deployed `gradient_boosting` scores r² 0.34 there. Fitting a straight line per
#: store first, and letting the tree model only the distance from it, scores 0.83 —
#: the same estimator, on the same rows, with the part it cannot represent taken out.
#:
#: It fixes the classifier at the same time. Demand terciles cut on raw sales drift to
#: 40.6% High in the held-out period against 33.3% in training, purely because the chain
#: grew; cut on the residual they stay near a third and go on meaning *busy for this shop,
#: now*.
#:
#: `--no-detrend` turns it off, which is how the two tables in the README were produced.
DETREND = True


#: The two problems the package solves. Named here because both `datasets` and
#: `registry` key off it and neither should own the vocabulary.
Task = Literal["regression", "classification"]
