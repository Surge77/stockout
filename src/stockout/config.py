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

# Six weeks. Long enough that lag-1 is unavailable, which is the entire point —
# a horizon of 1 lets you cheat with yesterday's sales and learn nothing.
DEFAULT_HORIZON_DAYS = 42
DEFAULT_N_FOLDS = 5
DEFAULT_GAP_DAYS = 0

# The shortest history a fold may train on. Two years, so every training window
# contains at least one full annual cycle plus a run-up.
DEFAULT_MIN_TRAIN_DAYS = 365

DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_REVIEW_PERIOD_DAYS = 7

# Newsvendor costs. Cu = cost of being short one unit (a lost sale), Co = cost of
# carrying one unit that did not sell. Their ratio, not a hyperparameter sweep,
# is what sets the target quantile — see inventory/policy.py::critical_ratio.
DEFAULT_UNDERAGE_COST = 3.0
DEFAULT_OVERAGE_COST = 1.0

# Weekly seasonality. Retail demand repeats on a 7-day cycle far more strongly than
# on any other, which is why the baseline to beat is same-weekday-last-week.
SEASON_LENGTH_DAYS = 7


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


def env_float(name: str, default: float) -> float:
    """Float counterpart to `env_int` — same fall-back-and-warn contract."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a valid float; using %s", name, raw, default)
        return default


def horizon_days() -> int:
    return env_int("STOCKOUT_HORIZON_DAYS", DEFAULT_HORIZON_DAYS)


def n_folds() -> int:
    return env_int("STOCKOUT_N_FOLDS", DEFAULT_N_FOLDS)


def gap_days() -> int:
    return env_int("STOCKOUT_GAP_DAYS", DEFAULT_GAP_DAYS)


def lead_time_days() -> int:
    return env_int("STOCKOUT_LEAD_TIME_DAYS", DEFAULT_LEAD_TIME_DAYS)


def review_period_days() -> int:
    return env_int("STOCKOUT_REVIEW_PERIOD_DAYS", DEFAULT_REVIEW_PERIOD_DAYS)


def underage_cost() -> float:
    return env_float("STOCKOUT_UNDERAGE_COST", DEFAULT_UNDERAGE_COST)


def overage_cost() -> float:
    return env_float("STOCKOUT_OVERAGE_COST", DEFAULT_OVERAGE_COST)


def load_env() -> None:
    """Load .env then .env.local (local values win). No-op if both are absent."""
    from dotenv import load_dotenv

    load_dotenv(".env")
    load_dotenv(".env.local", override=True)
