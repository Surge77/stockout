"""`stockout fetch | synth | describe | backtest | calibration | frontier`.

This module is the argument surface and nothing else: what each subcommand accepts, what
it defaults to, and which handler in `commands.py` receives it. Keeping the two apart
means a new flag is read next to the other flags rather than halfway down the function
that consumes it.

Handlers return an exit code and never raise past `main`; a domain error becomes a
one-line `error: ...` on stderr, because a traceback is not a user interface.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import config
from .commands import (
    run_backtest,
    run_calibration,
    run_describe,
    run_fetch,
    run_frontier,
    run_synth,
)
from .errors import StockoutError
from .models import FORECASTER_NAMES
from .models.conformal import ConformalQuantileForecaster
from .models.gbm import GbmQuantileForecaster


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252, which cannot encode the middle dot or the em
    dash this tool prints in every results line. Reconfigure rather than strip: a report
    that mangles its own punctuation invites doubt about the numbers next to it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stockout",
        description="Retail demand forecasting scored by the replenishment decision it drives.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download the Rossmann archive from Kaggle")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if present")

    p_synth = sub.add_parser("synth", help="generate a synthetic sales CSV")
    p_synth.add_argument("--out", type=Path, default=config.SAMPLE_PATH)
    p_synth.add_argument("--stores", type=int, default=4)
    p_synth.add_argument("--days", type=int, default=730)
    p_synth.add_argument("--seed", type=int, default=7)

    p_describe = sub.add_parser("describe", help="row counts, null profile, calendar gaps")
    p_describe.add_argument("--data", type=Path, default=config.SAMPLE_PATH)

    p_backtest = sub.add_parser("backtest", help="rolling-origin backtest of one model")
    p_backtest.add_argument("--data", type=Path, default=config.SAMPLE_PATH)
    p_backtest.add_argument("--model", default="seasonal_naive", choices=FORECASTER_NAMES)
    p_backtest.add_argument("--folds", type=int, default=config.DEFAULT_N_FOLDS)
    p_backtest.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_DAYS)
    p_backtest.add_argument("--gap", type=int, default=config.DEFAULT_GAP_DAYS)
    p_backtest.add_argument(
        "--min-train-days", type=int, default=config.DEFAULT_MIN_TRAIN_DAYS
    )
    p_backtest.add_argument(
        "--sliding",
        action="store_true",
        help="fixed-width training window instead of an expanding one",
    )

    p_frontier = sub.add_parser(
        "frontier", help="price each service level by the stock and lost sales it implies"
    )
    p_frontier.add_argument("--data", type=Path, default=config.SAMPLE_PATH)
    p_frontier.add_argument(
        "--store", type=int, default=None, help="defaults to the first store in the file"
    )
    p_frontier.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_DAYS)
    p_frontier.add_argument("--gap", type=int, default=config.DEFAULT_GAP_DAYS)
    p_frontier.add_argument(
        "--min-train-days", type=int, default=config.DEFAULT_MIN_TRAIN_DAYS
    )
    p_frontier.add_argument(
        "--review-period",
        type=int,
        default=config.DEFAULT_REVIEW_PERIOD_DAYS,
        help="cycle length for the service-level column; does not change what is ordered",
    )
    p_frontier.add_argument(
        "--lead-time",
        type=int,
        default=0,
        help="days between placing an order and its arrival; 0 prices a repeated "
        "newsvendor, above 0 opens a delivery pipeline and sizes across the "
        "protection interval",
    )
    p_frontier.add_argument(
        "--model",
        default=GbmQuantileForecaster.name,
        choices=(GbmQuantileForecaster.name, ConformalQuantileForecaster.name),
    )

    p_calibration = sub.add_parser(
        "calibration", help="does each quantile cover the share of days it claims to"
    )
    p_calibration.add_argument("--data", type=Path, default=config.SAMPLE_PATH)
    p_calibration.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_DAYS)
    p_calibration.add_argument("--gap", type=int, default=config.DEFAULT_GAP_DAYS)
    p_calibration.add_argument(
        "--min-train-days", type=int, default=config.DEFAULT_MIN_TRAIN_DAYS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    config.load_env()

    args = _parser().parse_args(argv)
    handlers = {
        "fetch": run_fetch,
        "synth": run_synth,
        "describe": run_describe,
        "backtest": run_backtest,
        "frontier": run_frontier,
        "calibration": run_calibration,
    }
    try:
        return handlers[args.command](args)
    except StockoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
