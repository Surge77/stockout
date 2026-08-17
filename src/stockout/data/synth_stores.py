"""Deterministic store metadata, so the preprocessing pipeline is testable offline.

`synth.py` reproduces the four structures that make *forecasting* Rossmann hard. This
module reproduces the four that make *preprocessing* it hard, which is a different list
and matters just as much now that a `ColumnTransformer` sits in the middle of the
package:

1. **Two categorical columns**, four and three levels — what `OneHotEncoder` is for.
2. **Numeric columns with genuine missing values**, in three distinct patterns — what
   `SimpleImputer` is for. A generator with no NaNs leaves the imputer untested while
   reporting full coverage of it, which is the worst of both.
3. **A text column.** `promo_interval` is a comma-separated month list, and the only
   text the dataset has — what `TfidfVectorizer` is for.
4. **Missingness that means something.** `promo2_since_week`, `promo2_since_year` and
   `promo_interval` are absent for exactly the stores not running the continuing
   promotion. That is not a data-quality problem to be patched; it is the value.

Assignment is by store index rather than by a random draw. A four-store fixture must
exercise every level and every NaN pattern, and a random choice over four stores would
usually miss two of them — a fixture that only sometimes covers the case is a fixture
that only sometimes tests it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schemas as s

#: Rossmann's four store types and three assortment levels, in its own spelling.
_STORE_TYPES: tuple[str, ...] = ("a", "b", "c", "d")
_ASSORTMENTS: tuple[str, ...] = ("a", "b", "c")

#: The three promotion calendars Rossmann actually ships. Note `Sept` rather than `Sep`
#: in the third — a real inconsistency in the source file, kept because a vectoriser
#: that has only ever seen tidy month names is not the one being tested.
_PROMO_INTERVALS: tuple[str, ...] = (
    "Jan,Apr,Jul,Oct",
    "Feb,May,Aug,Nov",
    "Mar,Jun,Sept,Dec",
)

#: Which store index gets which gap. Rossmann has three stores in 1115 with no
#: competitor distance and roughly a third with no competitor opening date. The first
#: of those is deliberately over-represented here: at the true rate a four-store
#: fixture would contain no missing distance at all, so the median-imputer branch would
#: go untested while reporting itself covered. A fixture that only sometimes exercises
#: a case is a fixture that only sometimes tests it. The proportion is wrong on
#: purpose; the pattern is what is being reproduced.
_NO_DISTANCE_EVERY = 4
_NO_COMPETITION_DATE_EVERY = 3

#: Every second store runs the continuing promotion. Roughly half of Rossmann does.
_PROMO2_EVERY = 2

_MIN_DISTANCE_M = 20
_MAX_DISTANCE_M = 75_000
_EARLIEST_COMPETITION_YEAR = 1900
_LATEST_COMPETITION_YEAR = 2015
_PROMO2_FIRST_YEAR = 2009
_PROMO2_LAST_YEAR = 2015
_WEEKS_IN_YEAR = 52


def make_stores(*, n_stores: int = 8, seed: int = 7) -> pd.DataFrame:
    """One row per store, in the canonical schema, with the real missingness patterns.

    Deterministic: the same `n_stores` and `seed` always produce a byte-identical frame,
    on the same contract as `synth.make_sales`.

    The frame is returned already canonicalised — snake_case names and the dtypes in
    `schemas.STORE_DTYPES` — because a caller that has to remember to canonicalise a
    generated frame will eventually forget, and the failure is a silent dtype change
    rather than an error.
    """
    if n_stores < 1:
        raise ValueError("n_stores must be at least 1")

    rng = np.random.default_rng(seed)
    rows = [_one_store(offset, rng) for offset in range(n_stores)]

    frame = pd.DataFrame(rows)
    for column, dtype in s.STORE_DTYPES.items():
        frame[column] = frame[column].astype(pd.api.types.pandas_dtype(dtype))
    return frame


def _one_store(offset: int, rng: np.random.Generator) -> dict[str, object]:
    """One store's metadata. `offset` is 0-based; `store` ids start at 1."""
    runs_promo2 = offset % _PROMO2_EVERY == 0

    return {
        s.STORE: offset + 1,
        s.STORE_TYPE: _STORE_TYPES[offset % len(_STORE_TYPES)],
        s.ASSORTMENT: _ASSORTMENTS[offset % len(_ASSORTMENTS)],
        s.COMPETITION_DISTANCE: _competition_distance(offset, rng),
        **_competition_opened(offset, rng),
        s.PROMO2: int(runs_promo2),
        **_promo2_started(offset, rng, runs_promo2=runs_promo2),
    }


def _competition_distance(offset: int, rng: np.random.Generator) -> float:
    """Metres to the nearest competitor, or NaN for the occasional store that has none.

    Right-skewed on purpose — a handful of stores are tens of kilometres from anything —
    because that skew is the reason `SimpleImputer` here uses the median rather than the
    mean. An imputer tested only on symmetric data cannot demonstrate why.
    """
    if offset % _NO_DISTANCE_EVERY == _NO_DISTANCE_EVERY - 1:
        return float("nan")
    return float(rng.integers(_MIN_DISTANCE_M, _MAX_DISTANCE_M) ** 0.85)


def _competition_opened(offset: int, rng: np.random.Generator) -> dict[str, float]:
    """Month and year the competitor opened. Absent together, never one without the other."""
    if offset % _NO_COMPETITION_DATE_EVERY == _NO_COMPETITION_DATE_EVERY - 1:
        return {
            s.COMPETITION_OPEN_MONTH: float("nan"),
            s.COMPETITION_OPEN_YEAR: float("nan"),
        }
    return {
        s.COMPETITION_OPEN_MONTH: float(rng.integers(1, 13)),
        s.COMPETITION_OPEN_YEAR: float(
            rng.integers(_EARLIEST_COMPETITION_YEAR, _LATEST_COMPETITION_YEAR + 1)
        ),
    }


def _promo2_started(
    offset: int, rng: np.random.Generator, *, runs_promo2: bool
) -> dict[str, object]:
    """When the continuing promotion began, and which months it repeats in.

    All three fields are missing together for a store not running it. That is the
    pattern worth reproducing: the absence is not damage, it is the answer to "does this
    store run Promo2", already encoded in `promo2`. An imputer that fills these with a
    median invents a promotion that does not exist.

    `promo_interval` stays NA here rather than becoming the empty string. Filling it is
    `loaders.merge_store`'s job, at the point where the column meets a vectoriser that
    cannot accept a null — doing it earlier would hide the pattern this module exists
    to produce.
    """
    if not runs_promo2:
        return {
            s.PROMO2_SINCE_WEEK: float("nan"),
            s.PROMO2_SINCE_YEAR: float("nan"),
            s.PROMO_INTERVAL: pd.NA,
        }
    return {
        s.PROMO2_SINCE_WEEK: float(rng.integers(1, _WEEKS_IN_YEAR + 1)),
        s.PROMO2_SINCE_YEAR: float(rng.integers(_PROMO2_FIRST_YEAR, _PROMO2_LAST_YEAR + 1)),
        s.PROMO_INTERVAL: _PROMO_INTERVALS[offset % len(_PROMO_INTERVALS)],
    }
