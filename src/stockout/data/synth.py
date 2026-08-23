"""A deterministic stand-in for Rossmann, so nothing under tests/ needs a network.

The generator is not trying to be realistic. It is trying to reproduce the four
structures that make the real problem hard, and nothing else:

1. Weekly seasonality wide enough to dominate the noise, so a model that finds the
   weekday effect scores visibly better than one that does not.
2. Sunday and Christmas closures, so the open=0 -> sales=0 invariant is exercised.
3. A refurbishment gap — one store's rows *vanish* for a quarter and come back. Note
   that this is absence, not a run of zeros: Rossmann's refurbished stores are simply
   missing from the file, and code that assumes a contiguous calendar per store breaks
   on them.
4. A level shift, so a model that assumes stationarity is visibly punished.
5. **Autocorrelated residuals.** What is left after the calendar is explained is not
   independent day to day: weather, local events and footfall trends persist for a few
   days at a time. This one was added last and for a specific reason — without it the
   generator cannot demonstrate leakage *of any kind*.

   The argument is short and worth keeping. A leak works by letting a model see a
   neighbouring observation that is informative about the one it is predicting. If the
   only unpredictable component is i.i.d. noise then no neighbour is informative, the
   leak transmits nothing, and both arms of an experiment score the same — so the
   experiment cannot tell a good split from a bad one.

   So the residual is an AR(1) process. Its correlation at one day is `_NOISE_RHO` and at
   seven days is `_NOISE_RHO ** 7`, which is near zero. Calendar-adjacent rows are
   therefore worth something and week-apart rows are not, which is exactly the asymmetry
   a shuffled split exploits and a chronological one does not. `stockout.leakage` measures
   that gap; this structure is why there is anything there to measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas as s

#: Monday is busiest and Saturday second; midweek sags. Indexed 1..7 to match
#: Rossmann's DayOfWeek rather than pandas' 0..6.
#:
#: The spread is wide on purpose. Real Rossmann stores swing this much between Monday
#: and Thursday, and if the synthetic weekday effect were smaller than the noise then
#: `SeasonalNaive` and `NaiveLast` would score the same here and differ on real data —
#: the test suite would be asserting something that is only true of the generator.
_DOW_MULTIPLIER: dict[int, float] = {1: 1.32, 2: 0.96, 3: 0.88, 4: 0.86, 5: 1.06, 6: 1.22, 7: 0.0}

_PROMO_LIFT = 1.27
_ANNUAL_AMPLITUDE = 0.12
_NOISE_SD = 0.08
_AVG_BASKET = 9.5

#: How a store's own attributes set the level it trades at.
#:
#: This used to be `rng.integers(3_500, 9_000)` — a store's size drawn at random, in a
#: different function from the one that generated its metadata. Nothing tied an attribute
#: to the level it was supposed to explain, and it showed: `competition_distance`
#: correlated with mean sales at r = -0.13 and assortment moved the mean by 8%, both
#: indistinguishable from noise. A model could only learn a store's level by memorising
#: *which store it was*, `entity_code` alone reached r² 0.49, and holding out a whole
#: store gave r² -0.70 — worse than guessing that store's own average.
#:
#: That is what made the package behave like a lookup table rather than a model. The
#: factors below give the metadata real explanatory power, in the direction retail
#: actually works: a wider range sells more, a competitor next door hurts, and the
#: continuing promotion helps a little.
_BASE_LEVEL = 6_000.0

#: Rossmann's documented assortment codes: basic, extra, extended. More range, more sales.
_ASSORTMENT_FACTOR: dict[str, float] = {"a": 0.85, "b": 1.00, "c": 1.22}

#: Store type is anonymised in the real data, so these are arbitrary but *consistent* —
#: the point is that the column carries signal, not that b means anything in particular.
_STORE_TYPE_FACTOR: dict[str, float] = {"a": 1.12, "b": 0.94, "c": 1.00, "d": 0.90}

_PROMO2_FACTOR = 1.08

#: Competition is felt on a log scale and saturates: the difference between 50 m and
#: 500 m matters, the difference between 40 km and 50 km does not. Distances are measured
#: against a 3 km pivot, and the effect is bounded so no single store is destroyed by it.
_COMPETITION_PIVOT_M = 3_000.0
_COMPETITION_SENSITIVITY = 0.16
_COMPETITION_MIN_FACTOR = 0.78
_COMPETITION_MAX_FACTOR = 1.22

#: What the attributes do **not** explain: location, management, passing trade. Kept
#: deliberately — without it the metadata would determine the level exactly, a held-out
#: store would score as well as a seen one, and the split between "predict a shop I know"
#: and "predict a shop I have never seen" would stop being a real distinction.
_UNEXPLAINED_SD = 0.10

#: Day-to-day persistence of the residual. 0.7 makes yesterday genuinely informative
#: about today — correlation 0.7 at one day — while leaving a week-old value nearly
#: useless, since 0.7 ** 7 is about 0.08. That asymmetry is the whole point: it is what
#: separates a horizon-respecting lag from a leaking one, and without it the two are
#: indistinguishable. See structure 5 in the module docstring.
_NOISE_RHO = 0.7

#: One store closes for refurbishment. Chosen as an index into the store list so a
#: caller asking for two stores still gets the behaviour.
_REFURB_STORE_INDEX = 1
_REFURB_START_DAY = 400
_REFURB_LENGTH_DAYS = 90

#: A different store steps up permanently — a competitor closing, say.
_SHIFT_STORE_INDEX = 2
_SHIFT_START_DAY = 500
_SHIFT_FACTOR = 1.30


def make_sales(
    *,
    n_stores: int = 8,
    start: str = "2013-01-01",
    days: int = 730,
    seed: int = 7,
    stores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a sales frame with the canonical schema.

    Deterministic: the same `seed`, `n_stores`, `start` and `days` always produce a
    byte-identical frame. `scripts/make_sample.py` relies on that to regenerate the
    committed sample without producing a spurious diff.

    `stores` is the metadata frame the levels are derived from. Passing the same frame
    that gets written beside the sales file is what keeps the two consistent; omitting it
    regenerates the equivalent frame from the same seed.
    """
    if n_stores < 1:
        raise ValueError("n_stores must be at least 1")
    if days < 1:
        raise ValueError("days must be at least 1")

    if stores is None:
        from .synth_stores import make_stores

        stores = make_stores(n_stores=n_stores, seed=seed)

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=days, freq="D")
    day_index = np.arange(days)
    dow = dates.dayofweek.to_numpy() + 1  # 1=Monday .. 7=Sunday

    # Promotions run Monday-Friday on alternating weeks, which is roughly Rossmann's
    # cadence and, more importantly, is not correlated with the noise.
    week = day_index // 7
    promo = ((week % 2 == 0) & (dow <= 5)).astype(np.int8)

    month, day = np.asarray(dates.month), np.asarray(dates.day)
    school_holiday = np.asarray(
        np.isin(month, [7, 8]) | ((month == 12) & (day >= 20))
    ).astype(np.int8)
    state_holiday = np.where((month == 12) & (day == 25), "a", "0")

    annual = 1.0 + _ANNUAL_AMPLITUDE * np.sin(2 * np.pi * day_index / 365.25)
    dow_factor = np.array([_DOW_MULTIPLIER[d] for d in dow])

    levels = _store_levels(stores, n_stores, rng)

    frames = []
    for offset in range(n_stores):
        store_id = offset + 1
        base = levels[offset]

        level = np.full(days, base)
        if offset == _SHIFT_STORE_INDEX:
            level = np.where(day_index >= _SHIFT_START_DAY, base * _SHIFT_FACTOR, base)

        is_open = (dow != s.SUNDAY).astype(np.int8)
        is_open = np.where(state_holiday == "a", 0, is_open).astype(np.int8)

        noise = _persistent_noise(rng, days)
        lift = np.where(promo == 1, _PROMO_LIFT, 1.0)
        sales = level * dow_factor * annual * lift * noise
        sales = np.where(is_open == 1, np.maximum(sales, 0.0).round(0), 0.0)

        customers = np.where(
            is_open == 1, np.maximum(sales / _AVG_BASKET + rng.normal(0, 8, days), 0).round(0), 0.0
        )

        store_frame = pd.DataFrame(
            {
                s.DATE: dates,
                s.STORE: np.int32(store_id),
                s.DAY_OF_WEEK: dow.astype(np.int8),
                s.SALES: sales.astype(np.float64),
                s.CUSTOMERS: customers.astype(np.float64),
                s.OPEN: is_open,
                s.PROMO: promo,
                s.STATE_HOLIDAY: pd.array(state_holiday, dtype="string"),
                s.SCHOOL_HOLIDAY: school_holiday,
            }
        )

        if offset == _REFURB_STORE_INDEX:
            # Rows are removed, not zeroed. The store is shut for refurbishment and
            # simply stops appearing, which is what Rossmann does and what breaks any
            # code assuming one row per store per day.
            refurb = (day_index >= _REFURB_START_DAY) & (
                day_index < _REFURB_START_DAY + _REFURB_LENGTH_DAYS
            )
            store_frame = store_frame[~refurb]

        frames.append(store_frame)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(list(s.KEY_COLUMNS)).reset_index(drop=True)


def _store_levels(stores: pd.DataFrame, n_stores: int, rng: np.random.Generator) -> list[float]:
    """The everyday trading level of each store, derived from its own metadata.

    One multiplicative factor per attribute, times an unexplained store effect. Returned
    in store-id order so `levels[offset]` belongs to store `offset + 1`.
    """
    indexed = stores.set_index(s.STORE)
    levels: list[float] = []

    for offset in range(n_stores):
        row = indexed.loc[offset + 1]
        level = (
            _BASE_LEVEL
            * _ASSORTMENT_FACTOR.get(str(row[s.ASSORTMENT]).lower(), 1.0)
            * _STORE_TYPE_FACTOR.get(str(row[s.STORE_TYPE]).lower(), 1.0)
            * _competition_factor(row[s.COMPETITION_DISTANCE])
            * (_PROMO2_FACTOR if _truthy(row[s.PROMO2]) else 1.0)
            * float(rng.lognormal(0.0, _UNEXPLAINED_SD))
        )
        levels.append(float(level))
    return levels


def _truthy(value: object) -> bool:
    """`1`/`0` out of a pandas row, without tripping over NA."""
    try:
        return bool(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _competition_factor(distance_m: object) -> float:
    """How much a nearby competitor costs this store.

    Log-scaled against `_COMPETITION_PIVOT_M` and bounded, because the effect saturates:
    50 m versus 500 m matters a great deal, 40 km versus 50 km does not.

    A missing distance is left neutral rather than treated as "no competitor nearby".
    The favourable reading is the more realistic one, but `features.py` imputes this
    column, so making absence advantageous would quietly turn the imputer into the thing
    under test. That belongs in its own experiment, not smuggled in here.
    """
    try:
        distance = float(distance_m)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(distance) or distance <= 0:
        return 1.0

    factor = 1.0 + _COMPETITION_SENSITIVITY * float(np.log10(distance / _COMPETITION_PIVOT_M))
    return float(np.clip(factor, _COMPETITION_MIN_FACTOR, _COMPETITION_MAX_FACTOR))


def _persistent_noise(rng: np.random.Generator, days: int) -> np.ndarray:
    """A multiplicative AR(1) residual with mean 1 and marginal SD `_NOISE_SD`.

    `sqrt(1 - rho**2)` on the innovations is what keeps the *marginal* standard deviation
    at `_NOISE_SD` rather than letting it inflate to `_NOISE_SD / sqrt(1 - rho**2)`.
    Without it, turning up the persistence would also turn up the volatility and the two
    effects could not be told apart.

    Written as a loop rather than a filter call: 730 days is nothing, and the recurrence
    is the definition of the thing.
    """
    innovation_sd = _NOISE_SD * float(np.sqrt(1.0 - _NOISE_RHO**2))
    innovations = rng.normal(0.0, innovation_sd, size=days)

    deviation = np.empty(days, dtype="float64")
    # Start from the stationary distribution, not from zero. Seeding at zero would make
    # every store unnaturally calm for its first fortnight.
    deviation[0] = rng.normal(0.0, _NOISE_SD)
    for day in range(1, days):
        deviation[day] = _NOISE_RHO * deviation[day - 1] + innovations[day]
    return 1.0 + deviation
