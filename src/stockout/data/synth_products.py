"""Generate a store-item sales panel, so the product path runs on a fresh clone.

The real product source is Kaggle's Store Item Demand data, which cannot be redistributed
and needs credentials. This is its committed stand-in, and it exists for the same reason
`synth.py` exists beside Rossmann: without it nothing on the product path is testable on a
machine that has never authenticated to Kaggle.

It mirrors the real file's shape exactly — `date, store, item, sales`, no metadata, no
promotions, no holidays, no closed days — so both adapters share one code path and a test
written against this one is a test of the real one.

**Store and item effects are generated separately, and that is the point.** Demand here is
a product of an item's own level, a store's own multiplier and the calendar:

    units  ~  Poisson( base[item] x multiplier[store] x weekday x season x growth )

A model given only the composite series id has to learn all 200 of those levels
independently. A model given `store_code` and `item_code` as separate features can learn
twenty item levels and ten store multipliers and compose them, which is fewer quantities
estimated from more rows each. `features.PRODUCT_FEATURES` exists to make that possible
and this generator is what makes the difference measurable rather than asserted.

**Counts are Poisson, not Gaussian.** Units sold are a count: non-negative, integer, and
noisier in absolute terms where the level is higher. Gaussian noise on a count produces
negative sales for the quiet items and understates the spread for the busy ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas as s

#: The real file is 10 stores x 50 items over five years. The committed sample keeps the
#: store count and trims the catalogue, spending its row budget on calendar depth instead:
#: two full years show the yearly season and leave every series more than
#: `trend.MIN_ROWS_PER_ENTITY` training rows after a chronological split, where a wider
#: catalogue over a few months would leave every series falling back to the chain line.
DEFAULT_STORES = 10
DEFAULT_ITEMS = 20
DEFAULT_DAYS = 730

#: The catalogue: what each item *is*, and how briskly its kind of thing sells.
#:
#: The real Store Item Demand file publishes an integer and nothing else — no name, no
#: category, no price — so this is the one place the two sources deliberately differ. That
#: is not an invented label for anonymised data, which this project refuses to do elsewhere
#: (Rossmann's shop groups stay letters); it is a fixture being authored. `store_item_demand`
#: keeps its bare integers and declares `has_item_metadata=False`.
#:
#: The category carries a **base rate**, so what an item is genuinely predicts how much of
#: it sells — the same correction `data/synth.py` needed when a store's level was drawn
#: independently of its attributes and nothing tied the two together. Without it `category`
#: would be a column the model could not use and the described-item page could not answer.
PRODUCT_CATALOGUE: tuple[tuple[str, str, float], ...] = (
    # (name, category, average units a day in a mid-sized shop)
    ("Kitchen towel, 4 pack", "household", 52.0),
    ("Washing-up liquid, 500ml", "household", 34.0),
    ("Bin bags, 20 pack", "household", 28.0),
    ("Surface spray, 750ml", "household", 19.0),
    ("Still water, 6x1.5L", "drinks", 47.0),
    ("Cola, 8x330ml", "drinks", 41.0),
    ("Orange juice, 1L", "drinks", 26.0),
    ("Instant coffee, 200g", "drinks", 17.0),
    ("Crisps, sharing bag", "snacks", 44.0),
    ("Milk chocolate, 100g", "snacks", 31.0),
    ("Mixed nuts, 200g", "snacks", 14.0),
    ("Shampoo, 400ml", "personal care", 23.0),
    ("Toothpaste, 100ml", "personal care", 21.0),
    ("Shower gel, 250ml", "personal care", 18.0),
    ("Hand soap, 300ml", "personal care", 12.0),
    ("Nappies, size 4", "baby", 11.0),
    ("Baby wipes, 3 pack", "baby", 9.0),
    ("Infant formula, 800g", "baby", 5.0),
    ("Moisturiser, 50ml", "beauty", 7.0),
    ("Lipstick", "beauty", 4.0),
)

#: The closed set of categories, in the order they appear above. Fixed rather than derived,
#: because `features` label-encodes it against a vocabulary that must not depend on which
#: rows a fold happened to contain.
PRODUCT_CATEGORIES: tuple[str, ...] = (
    "household", "drinks", "snacks", "personal care", "baby", "beauty",
)

#: How much an item's own level may stray from its category's base rate. Kept modest: the
#: category has to be worth something to a model, and an item-level multiplier wide enough
#: to swamp it would make `category` noise wearing a label.
_ITEM_SPREAD = 0.35

#: Smallest and largest shop, as a multiplier on any item's level. Deliberately a narrower
#: span than the item range: which item it is should matter more than which shop it is in,
#: which is what makes the two effects distinguishable rather than interchangeable.
_SMALLEST_STORE = 0.65
_LARGEST_STORE = 1.60

#: Monday to Sunday. Retail footfall climbs into the weekend and falls back on Monday.
_WEEKDAY_LIFT: tuple[float, ...] = (1.00, 0.95, 0.97, 1.02, 1.15, 1.35, 1.18)

#: Depth of the yearly cycle and the day it peaks. Day 200 is mid-July.
_SEASON_DEPTH = 0.25
_SEASON_PEAK_DAY = 200
_DAYS_PER_YEAR = 365.25

#: Yearly growth, drawn per series. Spread rather than a single rate because the whole
#: point of `trend.py` is that one chain-wide line will not do — and a generator where
#: every series grows identically would let a single global line score as well as 200
#: per-series ones, hiding the thing the decomposition is for.
_SLOWEST_GROWTH = 0.01
_FASTEST_GROWTH = 0.09

#: Below this an item is not stocked at all rather than merely quiet. Keeps a Poisson draw
#: from producing a series of mostly zeros that no model can say anything useful about.
_MIN_RATE = 0.5


def make_product_sales(
    *,
    n_stores: int = DEFAULT_STORES,
    n_items: int = DEFAULT_ITEMS,
    start: str = "2013-01-01",
    days: int = DEFAULT_DAYS,
    seed: int = 7,
) -> pd.DataFrame:
    """One row per store-item-day, in the real source's four columns.

    Deterministic for a given seed, so a run that produces a diff means this generator
    changed rather than that the sample drifted.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")

    catalogue = _catalogue_for(n_items)
    # Each item's level is its category's base rate, nudged. So `category` predicts demand
    # and an item still has an identity of its own worth knowing.
    item_base = np.asarray([rate for _, _, rate in catalogue], dtype="float64")
    item_base *= rng.uniform(1.0 - _ITEM_SPREAD, 1.0 + _ITEM_SPREAD, size=len(catalogue))

    store_multiplier = np.linspace(_SMALLEST_STORE, _LARGEST_STORE, n_stores)
    # Shuffled so a store's id is not its own ranking. Left in order, `store_code` would be
    # a monotone stand-in for the shop's size and one split would recover most of the
    # signal — which would flatter the feature this sample exists to demonstrate.
    rng.shuffle(store_multiplier)

    calendar = _calendar_factor(dates)
    elapsed_years = np.arange(days, dtype="float64") / _DAYS_PER_YEAR

    frames: list[pd.DataFrame] = []
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            growth = rng.uniform(_SLOWEST_GROWTH, _FASTEST_GROWTH)
            rate = (
                item_base[item - 1]
                * store_multiplier[store - 1]
                * calendar
                * (1.0 + growth * elapsed_years)
            )
            units = rng.poisson(np.maximum(rate, _MIN_RATE))
            frames.append(
                pd.DataFrame(
                    {
                        s.DATE: dates,
                        s.STORE: store,
                        s.ITEM: item,
                        s.SALES: units.astype("int64"),
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)


def _calendar_factor(dates: pd.DatetimeIndex) -> np.ndarray:
    """The weekday lift and the yearly season, as one multiplier per day.

    Both are cyclic, so they are the part a tree can learn and reuse on a date past the
    data — unlike the growth term, which is `trend.py`'s job precisely because no tree can
    reach a level it never saw.
    """
    weekday = np.asarray([_WEEKDAY_LIFT[day] for day in dates.dayofweek], dtype="float64")
    day_of_year = np.asarray(dates.dayofyear, dtype="float64")
    # Cosine, not sine: `cos` is at its maximum when the argument is zero, so the peak
    # lands on `_SEASON_PEAK_DAY` as the name promises. With `sin` the same expression
    # puts a zero crossing there and the real peak a quarter-year later — the constant
    # said mid-July and the data peaked in late October.
    season = 1.0 + _SEASON_DEPTH * np.cos(
        2.0 * np.pi * (day_of_year - _SEASON_PEAK_DAY) / _DAYS_PER_YEAR
    )
    return weekday * season


def _catalogue_for(n_items: int) -> list[tuple[str, str, float]]:
    """The first `n_items` catalogue entries, cycling if more are asked for than exist.

    Cycling rather than raising so the generator stays resizable from the command line;
    a second copy of an item is a different item id with the same description, which is
    what a real catalogue does with two pack sizes anyway.
    """
    catalogue = list(PRODUCT_CATALOGUE)
    while len(catalogue) < n_items:
        catalogue += list(PRODUCT_CATALOGUE)
    return catalogue[:n_items]


def make_catalogue(n_items: int = DEFAULT_ITEMS) -> pd.DataFrame:
    """One row per item: what it is, and which kind of thing it is.

    Written beside the sales file the way `sample_stores.csv` sits beside `sample_sales.csv`
    — the metadata a source publishes about its entities, kept out of the fact table rather
    than repeated across every one of its rows.
    """
    rows = [
        {s.ITEM: index, s.PRODUCT_NAME: name, s.CATEGORY: category}
        for index, (name, category, _) in enumerate(_catalogue_for(n_items), start=1)
    ]
    return pd.DataFrame(rows)
