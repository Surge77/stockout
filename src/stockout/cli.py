"""The argument surface: ten subcommands, grouped by what they are for.

This module is the argument surface and nothing else — what each subcommand accepts,
what it defaults to, and which handler in `commands.py` receives it. Keeping the two
apart means a new flag is read next to the other flags rather than halfway down the
function that consumes it.

`fetch`, `synth` and `describe` are about the data file. `prepare`, `backtest`,
`compare`, `leakage` and `tune` are about choosing a model. `train` and `predict` are
about serving the one that was chosen.

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
    run_compare,
    run_describe,
    run_fetch,
    run_leakage,
    run_predict,
    run_prepare,
    run_synth,
    run_train,
    run_tune,
)
from .errors import StockoutError
from .models import FORECASTER_NAMES
from .models.registry import model_names
from .persistence import artifact_path
from .train import DEFAULT_CLASSIFIER, DEFAULT_REGRESSOR

#: The default held-out window for `compare`, `leakage` and `train`. Four weeks: long
#: enough to span every weekday four times, short enough to leave the training window
#: nearly whole on a two-year file.
DEFAULT_TEST_DAYS = 28

#: Fewer folds than the backtest uses, matching `models/tuning.py`. A search costs
#: candidates x folds fits, and the honest number comes from the holdout afterwards.
DEFAULT_SEARCH_FOLDS = 3

_TASKS = ("regression", "classification")


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252, which cannot encode the middle dot or the em
    dash this tool prints in every results line. Reconfigure rather than strip: a report
    that mangles its own punctuation invites doubt about the numbers next to it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _add_frame_args(parser: argparse.ArgumentParser) -> None:
    """The three arguments every model-fitting command needs to build its frame.

    `--stores` is separate from `--data` because the real data is two files and the join
    is where four of the features come from. `dataset.prepare` refuses to proceed without
    it rather than quietly producing a frame missing its categorical branch.
    """
    parser.add_argument("--data", type=Path, default=config.SAMPLE_PATH)
    parser.add_argument("--stores", type=Path, default=config.SAMPLE_STORES_PATH)
    parser.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON_DAYS)


def _add_holdout_args(parser: argparse.ArgumentParser) -> None:
    """A held-out window, and the gap that stops it scoring a one-day forecast."""
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument(
        "--gap-days",
        type=int,
        default=None,
        help="days between the training and test windows (default: the horizon)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stockout",
        description="Retail demand forecasting, and the scikit-learn comparison behind it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    _add_data_commands(sub)
    _add_model_commands(sub)
    _add_serving_commands(sub)
    return parser


def _add_data_commands(sub: argparse._SubParsersAction) -> None:
    p_fetch = sub.add_parser("fetch", help="download the Rossmann archive from Kaggle")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if present")

    p_synth = sub.add_parser("synth", help="generate a synthetic sales CSV")
    p_synth.add_argument("--out", type=Path, default=config.SAMPLE_PATH)
    p_synth.add_argument("--stores", type=int, default=4)
    p_synth.add_argument("--days", type=int, default=730)
    p_synth.add_argument("--seed", type=int, default=7)

    p_describe = sub.add_parser("describe", help="row counts, null profile, calendar gaps")
    p_describe.add_argument("--data", type=Path, default=config.SAMPLE_PATH)

    p_prepare = sub.add_parser("prepare", help="build the model-ready frame and cache it")
    _add_frame_args(p_prepare)
    p_prepare.add_argument("--out", type=Path, default=config.PREPARED_CACHE)


def _add_model_commands(sub: argparse._SubParsersAction) -> None:
    p_backtest = sub.add_parser("backtest", help="rolling-origin backtest of one model")
    _add_frame_args(p_backtest)
    p_backtest.add_argument("--model", default="seasonal_naive", choices=FORECASTER_NAMES)
    p_backtest.add_argument("--folds", type=int, default=config.DEFAULT_N_FOLDS)
    p_backtest.add_argument("--gap", type=int, default=config.DEFAULT_GAP_DAYS)
    p_backtest.add_argument("--min-train-days", type=int, default=config.DEFAULT_MIN_TRAIN_DAYS)
    p_backtest.add_argument(
        "--sliding",
        action="store_true",
        help="fixed-width training window instead of an expanding one",
    )

    p_compare = sub.add_parser("compare", help="every registered model on one holdout")
    _add_frame_args(p_compare)
    _add_holdout_args(p_compare)
    p_compare.add_argument("--task", default="regression", choices=_TASKS)
    p_compare.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="a subset to score (default: the whole registry, simplest first)",
    )

    p_leakage = sub.add_parser("leakage", help="what four validation protocols believe")
    _add_frame_args(p_leakage)
    _add_holdout_args(p_leakage)

    p_tune = sub.add_parser("tune", help="grid search one or more models over time folds")
    _add_frame_args(p_tune)
    p_tune.add_argument("--task", default="regression", choices=_TASKS)
    p_tune.add_argument("--models", nargs="+", required=True)
    p_tune.add_argument("--folds", type=int, default=DEFAULT_SEARCH_FOLDS)


def _add_serving_commands(sub: argparse._SubParsersAction) -> None:
    p_train = sub.add_parser("train", help="fit both tasks on everything and save one file")
    _add_frame_args(p_train)
    _add_holdout_args(p_train)
    p_train.add_argument(
        "--regressor", default=DEFAULT_REGRESSOR, choices=model_names("regression")
    )
    p_train.add_argument(
        "--classifier", default=DEFAULT_CLASSIFIER, choices=model_names("classification")
    )
    p_train.add_argument("--out", type=Path, default=None)

    p_predict = sub.add_parser("predict", help="forecast one store-day from a saved model")
    p_predict.add_argument("--data", type=Path, default=config.SAMPLE_PATH)
    p_predict.add_argument("--stores", type=Path, default=config.SAMPLE_STORES_PATH)
    p_predict.add_argument("--model-path", type=Path, default=artifact_path())
    p_predict.add_argument("--store", type=int, required=True)
    p_predict.add_argument("--date", required=True, help="the day to forecast, YYYY-MM-DD")
    p_predict.add_argument("--promo", type=int, default=0, choices=(0, 1))
    p_predict.add_argument("--school-holiday", type=int, default=0, choices=(0, 1))
    p_predict.add_argument("--state-holiday", default="0")
    p_predict.add_argument("--closed", action="store_true", help="the store is shut that day")


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    config.load_env()

    args = _parser().parse_args(argv)
    handlers = {
        "fetch": run_fetch,
        "synth": run_synth,
        "describe": run_describe,
        "prepare": run_prepare,
        "backtest": run_backtest,
        "compare": run_compare,
        "leakage": run_leakage,
        "tune": run_tune,
        "train": run_train,
        "predict": run_predict,
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
