"""`stockout fetch | synth | describe | backtest`.

Handlers return an exit code and never raise past `main`; a domain error becomes a
one-line `error: ...` on stderr, because a traceback is not a user interface.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import config
from .data import schemas as s
from .data.loaders import read_sales, write_sales
from .data.synth import make_sales
from .data.validate import calendar_gaps, null_profile, validate_sales
from .errors import StockoutError
from .evaluate.backtest import backtest
from .evaluate.report import to_markdown
from .models.baselines import BASELINES


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
    p_backtest.add_argument("--model", default="seasonal_naive", choices=sorted(BASELINES))
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
    return parser


def _fetch(args: argparse.Namespace) -> int:
    from .data.download import fetch

    target = fetch(force=args.force)
    print(f"archive ready in {target}")
    return 0


def _synth(args: argparse.Namespace) -> int:
    frame = make_sales(n_stores=args.stores, days=args.days, seed=args.seed)
    validate_sales(frame)
    path = write_sales(frame, args.out)
    print(f"wrote {len(frame):,} rows across {args.stores} stores to {path}")
    return 0


def _describe(args: argparse.Namespace) -> int:
    frame = read_sales(args.data)
    validate_sales(frame)

    trading = frame[frame[s.OPEN] == 1]
    print(f"rows          {len(frame):,}")
    print(f"stores        {frame[s.STORE].nunique():,}")
    print(f"dates         {frame[s.DATE].min().date()} to {frame[s.DATE].max().date()}")
    print(f"trading days  {len(trading):,} ({len(trading) / max(len(frame), 1):.1%} of rows)")
    print(f"mean sales    {trading[s.SALES].mean():,.0f} (trading days only)")

    print("\nnulls")
    print(null_profile(frame).to_string(index=False))

    gaps = calendar_gaps(frame)
    print(f"\ncalendar gaps: {len(gaps)} store(s) with missing days")
    if not gaps.empty:
        print(gaps.to_string(index=False))
    return 0


def _backtest(args: argparse.Namespace) -> int:
    frame = read_sales(args.data)
    validate_sales(frame)

    factory = BASELINES[args.model]
    results = backtest(
        frame,
        factory,
        n_folds=args.folds,
        horizon=args.horizon,
        gap=args.gap,
        min_train_days=args.min_train_days,
        expanding=not args.sliding,
    )
    print(to_markdown(results, model_name=args.model))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    config.load_env()

    args = _parser().parse_args(argv)
    handlers = {
        "fetch": _fetch,
        "synth": _synth,
        "describe": _describe,
        "backtest": _backtest,
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
