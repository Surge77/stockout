"""One function that turns two CSVs into the frame every model expects.

Six steps have to happen in one order, and until now every caller repeated them: read the
sales file, join the store metadata, derive the date-relative store features, build the
calendar and lag features, sort date-major, drop the warm-up rows the lags cannot fill.
Get the order wrong and the failure is quiet — `add_store_features` before the join
silently produces nothing, `date_major` before the lags builds them across store
boundaries.

So the order lives here once, and `notebooks/`, the CLI and the web app all call the same
function. A pipeline that is written down in four places is four pipelines.

**The warm-up drop is not optional.** `sales_lag_28` is null for a store's first 28 rows
and `sales_roll_mean_91` for its first 91. Handing those to an imputer would fill them
with the median of a column that does not apply yet, which is worse than dropping them —
it invents history. The rows are dropped and the count is reported.

**Caching is parquet, not CSV.** Rossmann prepared is roughly 800k rows by fifty columns.
CSV is ~400 MB and re-parses dtypes on every read; parquet is a tenth of that and keeps
the categorical and nullable-integer dtypes the rest of the package relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config
from .data import schemas as s
from .data.loaders import merge_store, read_sales, read_store
from .data.validate import validate_sales
from .errors import SchemaError
from .features.build import build_features
from .features.store_features import add_store_features
from .split.strategies import date_major
from .targets import DemandThresholds, add_demand_class, fit_thresholds

#: The lag whose warm-up decides how many rows are lost. Always the longest rolling
#: window, because it is null for more rows than any single lag.
_WARMUP_COLUMN_TEMPLATE = "{column}_roll_mean_{window}"


@dataclass(frozen=True)
class Prepared:
    """A model-ready frame, and the facts about how it was made."""

    frame: pd.DataFrame
    horizon: int
    rows_in: int
    rows_dropped: int
    thresholds: DemandThresholds

    @property
    def rows(self) -> int:
        return len(self.frame)

    def summary(self) -> str:
        return (
            f"{self.rows:,} rows ready at horizon {self.horizon} "
            f"({self.rows_dropped:,} dropped for lag warm-up, {self.rows_in:,} read); "
            f"{self.frame[s.STORE].nunique():,} stores; "
            f"thresholds fitted for {self.thresholds.stores:,}"
        )


def prepare(
    *,
    sales_path: Path | str | None = None,
    stores_path: Path | str | None = None,
    horizon: int | None = None,
    label: bool = True,
) -> Prepared:
    """Read, join, featurise, order and label. The one path into a model.

    `label` fits the demand-class thresholds on the *whole* frame, which is correct for
    exploration and wrong for scoring — a threshold is a statistic, and one fitted over
    the test window has let the test window vote on its own labels. Anything that scores
    a model refits them on its training slice; `evaluate/comparison.py` does exactly that
    and says so.
    """
    horizon = horizon or config.horizon_days()
    sales_path = Path(sales_path or config.SAMPLE_PATH)
    stores_path = Path(stores_path or config.SAMPLE_STORES_PATH)

    sales = read_sales(sales_path)
    validate_sales(sales)
    rows_in = len(sales)

    if not stores_path.exists():
        # Raise rather than carry on without the join. Skipping it silently produces a
        # frame that looks fine and is missing the categorical, text and competition
        # features — every model still fits, scores a little worse, and nothing says why.
        hint = (
            " Run `python scripts/make_sample.py` to regenerate the committed samples."
            if stores_path == config.SAMPLE_STORES_PATH
            else ""
        )
        raise SchemaError(f"store metadata is missing at {stores_path}.{hint}")
    sales = merge_store(sales, read_store(stores_path))

    built = build_features(add_store_features(sales), horizon=horizon)
    ordered = date_major(built)
    ready = _drop_warmup(ordered)

    thresholds = fit_thresholds(ready) if label else DemandThresholds()
    if label:
        ready = add_demand_class(ready, thresholds)

    return Prepared(
        frame=ready,
        horizon=horizon,
        rows_in=rows_in,
        rows_dropped=rows_in - len(ready),
        thresholds=thresholds,
    )


def _drop_warmup(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the leading rows per store that no lag or rolling window can fill.

    Keyed on the longest rolling mean, which is null for more rows than any lag is. Every
    other generated column is filled wherever this one is.
    """
    longest = max(_rolling_windows(frame), default=0)
    column = _WARMUP_COLUMN_TEMPLATE.format(column=s.SALES, window=longest)
    if column not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.dropna(subset=[column]).reset_index(drop=True)


def _rolling_windows(frame: pd.DataFrame) -> list[int]:
    prefix = f"{s.SALES}_roll_mean_"
    return [int(name.removeprefix(prefix)) for name in frame.columns if name.startswith(prefix)]


def save_prepared(prepared: Prepared, path: Path | str) -> Path:
    """Cache a prepared frame so a notebook never rebuilds it. Parquet, not CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared.frame.to_parquet(destination, index=False)
    return destination


def load_prepared(path: Path | str, *, horizon: int | None = None) -> pd.DataFrame:
    """Read a cached frame back. The horizon is the caller's to remember.

    Deliberately not wrapped back into a `Prepared`: the row counts and thresholds
    describe a build that already happened, and reconstructing them from the cache would
    be inventing provenance rather than recording it.
    """
    frame = pd.read_parquet(Path(path))
    if horizon is not None:
        expected = f"{s.SALES}_lag_{horizon}"
        if expected not in frame.columns:
            raise SchemaError(
                f"{path} was not prepared at horizon {horizon}: {expected!r} is missing"
            )
    return frame
