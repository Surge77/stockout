"""A deterministic stand-in for Rossmann, so nothing under tests/ needs a network.

The generator is not trying to be realistic. It is trying to reproduce the four
structures that make the real problem hard, and nothing else:

1. Weekly seasonality strong enough that same-weekday-last-week is a serious
   baseline. If the synthetic data lacked it, `SeasonalNaive` would look bad here
   and good on Rossmann, and the test suite would be lying.
2. Sunday closures, so the open=0 -> sales=0 invariant is exercised.
3. A refurbishment gap — one store's rows *vanish* for a quarter and come back. Note
   that this is absence, not a run of zeros: Rossmann's refurbished stores are simply
   missing from the file, and code that assumes a contiguous calendar per store breaks
   on them. `validate.calendar_gaps` exists to find exactly this.
4. A level shift, so a model that assumes stationarity is visibly punished.
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
) -> pd.DataFrame:
    """Build a sales frame with the canonical schema.

    Deterministic: the same `seed`, `n_stores`, `start` and `days` always produce a
    byte-identical frame. `scripts/make_sample.py` relies on that to regenerate the
    committed sample without producing a spurious diff.
    """
    if n_stores < 1:
        raise ValueError("n_stores must be at least 1")
    if days < 1:
        raise ValueError("days must be at least 1")

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

    frames = []
    for offset in range(n_stores):
        store_id = offset + 1
        base = float(rng.integers(3_500, 9_000))

        level = np.full(days, base)
        if offset == _SHIFT_STORE_INDEX:
            level = np.where(day_index >= _SHIFT_START_DAY, base * _SHIFT_FACTOR, base)

        is_open = (dow != s.SUNDAY).astype(np.int8)
        is_open = np.where(state_holiday == "a", 0, is_open).astype(np.int8)

        noise = rng.normal(1.0, _NOISE_SD, size=days)
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
