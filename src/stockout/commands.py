"""What each subcommand actually does, once the parser has agreed on the arguments.

Split from `cli.py` so that the argument surface and the work are readable apart. The
parser's job is to reject a bad invocation before anything is loaded; these functions
assume that has happened and get on with it.

Every one of them returns an exit code and raises only `StockoutError` subclasses, which
`cli.main` turns into a one-line `error: ...`. A traceback is not a user interface.

**Two commands read the raw CSV and the rest read a prepared frame.** `describe` and
`synth` are about the file as it arrives, so they must not see engineered columns.
Everything that fits a model goes through `dataset.prepare`, which is the one place the
six preparation steps happen in the one order that is correct — see its docstring for
what goes wrong when a caller reimplements them.
"""

from __future__ import annotations

import argparse

from .data import schemas as s
from .data.loaders import read_sales, write_sales
from .data.synth import make_sales
from .data.validate import calendar_gaps, null_profile, validate_sales
from .dataset import Prepared, prepare, save_prepared
from .evaluate.backtest import backtest
from .evaluate.comparison import compare
from .evaluate.comparison import to_markdown as comparison_to_markdown
from .evaluate.leakage import leakage_arms
from .evaluate.report import frame_to_markdown, to_markdown
from .models import forecaster
from .models.tuning import tune_many
from .persistence import load, save
from .predict import forecast
from .train import train


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


def run_prepare(args: argparse.Namespace) -> int:
    """Build the model-ready frame once and cache it, so nothing rebuilds it per run."""
    prepared = _prepared(args)
    print(prepared.summary())
    path = save_prepared(prepared, args.out)
    print(f"cached to {path}")
    return 0


def run_backtest(args: argparse.Namespace) -> int:
    prepared = _prepared(args)
    results = backtest(
        prepared.frame,
        forecaster(args.model, horizon=prepared.horizon),
        n_folds=args.folds,
        horizon=prepared.horizon,
        gap=args.gap,
        min_train_days=args.min_train_days,
        expanding=not args.sliding,
    )
    print(to_markdown(results, model_name=args.model))
    return 0


def run_compare(args: argparse.Namespace) -> int:
    prepared = _prepared(args)
    table = compare(
        prepared.frame,
        task=args.task,
        test_days=args.test_days,
        gap_days=_gap_days(args, prepared),
        models=args.models,
    )
    print(comparison_to_markdown(table, task=args.task))
    return 0


def run_leakage(args: argparse.Namespace) -> int:
    prepared = _prepared(args)
    arms = leakage_arms(
        prepared.frame, test_days=args.test_days, gap_days=_gap_days(args, prepared)
    )
    print(frame_to_markdown(arms))
    print(
        "\n`optimism` is what the protocol over-reported: its own estimate minus what "
        "the model then scored on a window nobody trained on."
    )
    return 0


def run_tune(args: argparse.Namespace) -> int:
    prepared = _prepared(args)
    table = tune_many(
        prepared.frame,
        args.models,
        task=args.task,
        horizon=prepared.horizon,
        folds=args.folds,
    )
    print(frame_to_markdown(table))
    return 0


def run_train(args: argparse.Namespace) -> int:
    prepared = _prepared(args)
    artifact = train(
        prepared.frame,
        horizon=prepared.horizon,
        regressor=args.regressor,
        classifier=args.classifier,
        test_days=args.test_days,
        gap_days=_gap_days(args, prepared),
    )
    path = save(artifact, args.out)
    print(artifact.summary())
    print(f"wrote {path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    """Load a trained artifact and answer one store-day.

    The horizon comes from the artifact rather than from a flag. A model fitted at
    horizon 7 reads `sales_lag_7`, and preparing its history at any other horizon
    produces a frame whose columns it cannot find — see `persistence.py`.
    """
    artifact = load(args.model_path)
    prepared = prepare(
        sales_path=args.data, stores_path=args.stores, horizon=artifact.horizon
    )
    answer = forecast(
        artifact,
        prepared.frame,
        store=args.store,
        date=args.date,
        promo=args.promo,
        school_holiday=args.school_holiday,
        state_holiday=args.state_holiday,
        is_open=0 if args.closed else 1,
    )
    print(answer.summary())
    print(f"model: {artifact.summary()}")
    return 0


def _prepared(args: argparse.Namespace) -> Prepared:
    return prepare(
        sales_path=args.data, stores_path=args.stores, horizon=args.horizon
    )


def _gap_days(args: argparse.Namespace, prepared: Prepared) -> int:
    """Default the train/test gap to the horizon rather than to zero.

    A holdout whose training rows end the day before its test rows begin is scoring a
    one-day forecast however long the horizon says it is. `--gap-days` overrides, and
    passing 0 is a deliberate choice a reader can see in the command line.
    """
    return prepared.horizon if args.gap_days is None else args.gap_days
