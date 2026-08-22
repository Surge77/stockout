"""The argument surface: five subcommands, and which handler in `commands.py` gets each.

`describe` is about the data. `models` and `compare` are about choosing a model.
`train` and `predict` are about serving the one that was chosen.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .commands import (
    run_compare,
    run_describe,
    run_models,
    run_predict,
    run_train,
)
from .config import RANDOM_SEED, TEST_SIZE
from .datasets import DATASET_NAMES
from .errors import StockoutError
from .registry import TASK_NAMES


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252, which cannot encode the middle dot this tool
    prints in every results line. Reconfigure rather than strip: a report that mangles
    its own punctuation invites doubt about the numbers next to it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="salary")


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=TASK_NAMES, default="regression")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockout", description="scikit-learn regression and classification, as taught"
    )
    parser.add_argument("--version", action="version", version=f"stockout {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_describe = sub.add_parser("describe", help="row counts, dtypes, nulls, summary stats")
    _add_data_args(p_describe)
    p_describe.set_defaults(handler=run_describe)

    p_models = sub.add_parser("models", help="what --model accepts, per task")
    p_models.set_defaults(handler=run_models)

    p_compare = sub.add_parser("compare", help="every registered model on one split")
    _add_data_args(p_compare)
    _add_split_args(p_compare)
    p_compare.set_defaults(handler=run_compare)

    p_train = sub.add_parser("train", help="fit one model and save it")
    _add_data_args(p_train)
    _add_split_args(p_train)
    p_train.add_argument("--model", default=None, help="default: lasso / logistic")
    p_train.set_defaults(handler=run_train)

    p_predict = sub.add_parser("predict", help="one prediction from the saved model")
    p_predict.add_argument(
        "--interactive", action="store_true", help="prompt for each feature"
    )
    p_predict.add_argument(
        "--set", action="append", metavar="NAME=VALUE", help="repeatable feature value"
    )
    p_predict.set_defaults(handler=run_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_output()
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except StockoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
