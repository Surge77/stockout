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

#: Quietest and busiest item, in units on an average day. Log-spaced between the two, so
#: the catalogue has a few sellers and a long tail rather than twenty near-identical items.
_QUIETEST_ITEM = 4.0
_BUSIEST_ITEM = 55.0

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

    item_base = np.geomspace(_QUIETEST_ITEM, _BUSIEST_ITEM, n_items)
    store_multiplier = np.linspace(_SMALLEST_STORE, _LARGEST_STORE, n_stores)

    # Shuffled so that neither id is its own ranking. Left in order, `item_code` would be
    # a monotone stand-in for the item's level and a single split on it would recover most
    # of the signal — which would flatter the feature this sample exists to demonstrate.
    rng.shuffle(item_base)
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
    season = 1.0 + _SEASON_DEPTH * np.sin(
        2.0 * np.pi * (day_of_year - _SEASON_PEAK_DAY) / _DAYS_PER_YEAR
    )
    return weekday * season
