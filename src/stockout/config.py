"""Paths, forecasting defaults, and the cost pair the service level is derived from.

Env knobs are read lazily at the call site and never captured into module-level
constants at import time, so a test can `monkeypatch.setenv` without reimporting.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = _ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = _ROOT / "reports"
SAMPLE_PATH = DATA_DIR / "sample_sales.csv"

KAGGLE_COMPETITION = "rossmann-store-sales"

# One week. A supplier reorders weekly, so this is the horizon the business actually
# has. It is also long enough that lag-1 is unavailable, which is the point of having
# a horizon at all — at a horizon of 1 you can cheat with yesterday's sales and learn
# nothing. `features/lags.py` raises when a lag is shorter than this.
DEFAULT_HORIZON_DAYS = 7
DEFAULT_N_FOLDS = 5
DEFAULT_GAP_DAYS = 0

# The shortest history a fold may train on. A year, so every training window
# contains at least one full annual cycle.
DEFAULT_MIN_TRAIN_DAYS = 365

# Weekly seasonality. Retail demand repeats on a 7-day cycle far more strongly than
# on any other, which is why the baseline to beat is same-weekday-last-week.
SEASON_LENGTH_DAYS = 7

# Every estimator that draws random numbers is given this, so two runs of the same
# command produce the same table. Reproducibility is not the same thing as having no
# randomness — the previous version of this package banned seeds outright, which
# solved the problem by not having the feature. ADR 0003.
RANDOM_SEED = 42

# Kernel and instance-based models do not scale: SVC and SVR are between O(n^2) and
# O(n^3) in the training rows, and KNN pays at predict time instead. Roughly 800k rows
# survive feature building, so they get a stratified sample of this size and the wall
# clock is printed next to the score. A subsample that is reported is a decision; one
# that is silent is a fiddle. ADR 0015.
SUBSAMPLE_ROWS = 5_000

# The three demand classes, in the order a human reads them. Stated explicitly because
# sklearn's LabelEncoder sorts alphabetically, which would map High -> 0 and Low -> 1
# and caption every confusion matrix wrongly while looking entirely plausible.
DEMAND_CLASS_LABELS: tuple[str, ...] = ("Low", "Medium", "High")


def env_int(name: str, default: int) -> int:
    """Read an int knob, falling back to `default` when unset or malformed.

    A typo in one knob (`STOCKOUT_N_FOLDS=5f`) must degrade to the default and warn,
    not kill a backtest that has already spent minutes fitting folds.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid int; using %d", name, raw, default)
        return default


def horizon_days() -> int:
    return env_int("STOCKOUT_HORIZON_DAYS", DEFAULT_HORIZON_DAYS)


def n_folds() -> int:
    return env_int("STOCKOUT_N_FOLDS", DEFAULT_N_FOLDS)


def gap_days() -> int:
    return env_int("STOCKOUT_GAP_DAYS", DEFAULT_GAP_DAYS)


def subsample_rows() -> int:
    return env_int("STOCKOUT_SUBSAMPLE_ROWS", SUBSAMPLE_ROWS)


def random_seed() -> int:
    return env_int("STOCKOUT_RANDOM_SEED", RANDOM_SEED)


def load_env() -> None:
    """Load .env then .env.local (local values win). No-op if both are absent."""
    from dotenv import load_dotenv

    load_dotenv(".env")
    load_dotenv(".env.local", override=True)
