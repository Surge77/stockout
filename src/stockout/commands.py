"""What each subcommand actually does, once the parser has agreed on the arguments.

Split from `cli.py` so that the argument surface and the work are readable apart. The
parser's job is to reject a bad invocation before anything is loaded; these functions
assume that has happened and get on with it.

Every one of them returns an exit code and raises only `StockoutError` subclasses, which
`cli.main` turns into a one-line `error: ...`. A traceback is not a user interface.
"""

from __future__ import annotations

import argparse

from .data import schemas as s
from .data.loaders import read_sales, write_sales
from .data.synth import make_sales
from .data.validate import calendar_gaps, null_profile, validate_sales
from .evaluate.backtest import backtest
from .evaluate.report import to_markdown
from .models import forecaster


def run_fetch(args: argparse.Namespace) -> int:
    from .data.download import fetch

    target = fetch(force=args.force)
    print(f"archive ready in {target}")
    return 0


def run_synth(args: argparse.Namespace) -> int:
    frame = make_sales(n_stores=args.stores, days=args.days, seed=args.seed)
    validate_sales(frame)
    path = write_sales(frame, args.out)
    print(f"wrote {len(frame):,} rows across {args.stores} stores to {path}")
    return 0


def run_describe(args: argparse.Namespace) -> int:
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


def run_backtest(args: argparse.Namespace) -> int:
    frame = read_sales(args.data)
    validate_sales(frame)

    results = backtest(
        frame,
        forecaster(args.model, horizon=args.horizon),
        n_folds=args.folds,
        horizon=args.horizon,
        gap=args.gap,
        min_train_days=args.min_train_days,
        expanding=not args.sliding,
    )
    print(to_markdown(results, model_name=args.model))
    return 0
