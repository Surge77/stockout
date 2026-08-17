"""What each subcommand actually does, once the parser has agreed on the arguments.

Split from `cli.py` so that the argument surface and the work are readable apart. The
parser's job is to reject a bad invocation before anything is loaded; these functions
assume that has happened and get on with it.

Every one of them returns an exit code and raises only `StockoutError` subclasses, which
`cli.main` turns into a one-line `error: ...`. A traceback is not a user interface.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .data import schemas as s
from .data.loaders import read_sales, write_sales
from .data.synth import make_sales
from .data.validate import calendar_gaps, null_profile, validate_sales
from .errors import BacktestError
from .evaluate.backtest import backtest
from .evaluate.metrics import coverage_by_segment, coverage_table
from .evaluate.report import (
    calibration_to_markdown,
    conditional_coverage_to_markdown,
    frontier_to_markdown,
    to_markdown,
)
from .inventory.frontier import frontier
from .inventory.policy import critical_ratio
from .models import forecaster
from .models.conformal import ConformalQuantileForecaster
from .models.gbm import GbmQuantileForecaster
from .models.protocols import QuantileModel
from .split.rolling import Fold, rolling_origin, split_frame


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


def _newest_fold(args: argparse.Namespace) -> tuple[Fold, pd.DataFrame, pd.DataFrame]:
    """Read, validate, and lay out the single most recent rolling-origin fold."""
    frame = read_sales(args.data)
    validate_sales(frame)

    fold = rolling_origin(
        frame[s.DATE],
        n_folds=1,
        horizon=args.horizon,
        gap=args.gap,
        min_train_days=args.min_train_days,
    )[-1]
    train, test = split_frame(frame, fold)
    return fold, train, test


def _segment_for(frame: pd.DataFrame, by: str) -> pd.Series:
    """The grouping `--by` names, as a column aligned to `frame`.

    `store` stays an integer so that ten sorts after nine rather than after one, which is
    the kind of detail that makes a 1,115-store table unreadable.
    """
    if by == "month":
        return frame[s.DATE].dt.strftime("%Y-%m")
    return frame[s.STORE]


def _quantile_model(name: str, *, horizon: int) -> QuantileModel:
    if name == ConformalQuantileForecaster.name:
        return ConformalQuantileForecaster(horizon=horizon)
    return GbmQuantileForecaster(horizon=horizon)


def run_frontier(args: argparse.Namespace) -> int:
    """Fit quantiles on the newest fold, then price what stocking to each would cost.

    One store, because inventory is held per store and averaging a fill rate across a
    quiet shop and a busy one describes neither of them.
    """
    # Checked before anything is fitted. `simulate` would reject them too, but only after
    # LightGBM has spent several seconds training a model nobody can use, and a ValueError
    # escaping `main` is a traceback rather than a message.
    if args.review_period < 1:
        raise BacktestError("--review-period must be at least 1 day")
    if args.lead_time < 0:
        raise BacktestError("--lead-time cannot be negative")
    if args.transit_holding_cost is not None and args.transit_holding_cost < 0:
        raise BacktestError("--transit-holding-cost cannot be negative")

    fold, train, test = _newest_fold(args)

    store = int(test[s.STORE].iloc[0]) if args.store is None else args.store
    rows = test[s.STORE] == store
    if not bool(rows.any()):
        raise BacktestError(f"store {store} has no rows in the test window")

    model = _quantile_model(args.model, horizon=args.horizon).fit(train)
    quantiles = model.predict_quantiles(test)

    underage, overage = config.underage_cost(), config.overage_cost()
    table = frontier(
        test.loc[rows, s.SALES],
        quantiles.loc[rows],
        lead_time_days=args.lead_time,
        review_period_days=args.review_period,
        holding_cost=overage,
        shortage_cost=underage,
        transit_holding_cost=args.transit_holding_cost,
    )

    transit_rate = overage if args.transit_holding_cost is None else args.transit_holding_cost
    system = (
        "single-period stocking"
        if args.lead_time == 0
        else f"{args.lead_time}d lead time, sized across a {args.lead_time + 1}d "
        f"protection interval, transit charged at {transit_rate:.1f}/unit/day"
    )
    target = critical_ratio(underage_cost=underage, overage_cost=overage)
    print(
        f"store {store} · {fold.test_start.date()} to {fold.test_end.date()} · "
        f"horizon {args.horizon}d · {args.model} · {system}, {args.review_period}d cycles"
    )
    print(
        f"newsvendor target quantile {target:.2f} "
        f"(Cu {underage:.1f} short, Co {overage:.1f} carried) — derived, not tuned"
    )
    print(f"quantile crossing on {model.crossing_rate:.1%} of rows, sorted before use\n")
    print(frontier_to_markdown(table))
    return 0


def run_calibration(args: argparse.Namespace) -> int:
    """Score the raw and the calibrated quantiles on the same held-out window.

    Both tables, always. Printing only the calibrated one would turn a measurement into
    an advertisement, and the size of the correction is the interesting number.
    """
    fold, train, test = _newest_fold(args)
    trading = test[s.OPEN] == 1
    actual = test.loc[trading, s.SALES]

    # Checked before either model is fitted. A window of nothing but closures scores every
    # level NaN, and a coverage table of NaN reads as a result rather than as an absence.
    if not bool(trading.any()):
        raise BacktestError(
            f"the test window {fold.test_start.date()} to {fold.test_end.date()} contains "
            "no trading day, so no coverage can be measured"
        )

    group_by = None if args.calibrate_by is None else s.STORE
    raw = GbmQuantileForecaster(horizon=args.horizon).fit(train)
    calibrated = ConformalQuantileForecaster(
        horizon=args.horizon, group_by=group_by, refit=not args.no_refit
    ).fit(train)

    print(
        f"{fold.test_start.date()} to {fold.test_end.date()} · horizon {args.horizon}d · "
        f"{int(trading.sum()):,} trading rows held out · "
        f"{calibrated.calibration_rows:,} rows in the calibration window\n"
    )
    if args.no_refit:
        print(
            "> Served without a refit: the offsets describe the estimator that produced "
            "them, so the split-conformal theorem applies to it. Coverage is still not "
            "proven — the theorem also needs the calibration and test rows to be "
            "exchangeable, and a test window that comes after a calibration window is "
            "not. One assumption remains where there were two. ADR 0013.\n"
        )
    segment = None if args.by is None else _segment_for(test.loc[trading], args.by)
    for model in (raw, calibrated):
        predicted = model.predict_quantiles(test).loc[trading]
        print(
            calibration_to_markdown(
                coverage_table(actual, predicted), model_name=model.name
            )
        )
        if segment is not None:
            print(
                conditional_coverage_to_markdown(
                    coverage_by_segment(actual, predicted, segment=segment),
                    model_name=model.name,
                )
            )

    if group_by is not None:
        print(
            f"> Per-{args.calibrate_by} calibration needs {calibrated.min_group_rows:,} "
            f"trading rows per group before an offset is an estimate rather than the worst "
            f"day that happened. {len(calibrated.group_offsets)} group(s) cleared it; "
            f"{len(calibrated.pooled_fallback_groups)} fell back to the pooled offset.\n"
        )

    if calibrated.saturated_quantiles:
        levels = ", ".join(f"{level:.2f}" for level in calibrated.saturated_quantiles)
        print(
            f"> Quantiles {levels} saturate: the calibration window holds too few rows "
            f"for the finite-sample correction to land anywhere but on the largest "
            f"residual in it. Their offsets are the worst day that happened, not an "
            f"estimate of a quantile."
        )
    return 0
