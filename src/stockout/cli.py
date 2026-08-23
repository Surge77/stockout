"""The argument surface, and which handler in `commands.py` gets each subcommand.

`datasets` and `fetch` are about the data. `models` and `compare` are about choosing a
model. `train` and `predict` are about serving the pair that was chosen.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .commands import (
    run_compare,
    run_datasets,
    run_describe,
    run_fetch,
    run_leakage,
    run_models,
    run_predict,
    run_train,
)
from .config import DEFAULT_SPLIT, DETREND, RANDOM_SEED, TEST_SIZE
from .data.adapters import SOURCE_NAMES
from .errors import StockoutError
from .registry import TASK_NAMES, model_names
from .train import DEFAULT_CLASSIFIER, DEFAULT_REGRESSOR

DEFAULT_DATASET = "synthetic"


def _force_utf8_output() -> None:
    """Windows consoles default to cp1252, which cannot encode the punctuation this tool
    prints in every results line. Reconfigure rather than strip: a report that mangles its
    own punctuation invites doubt about the numbers next to it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SOURCE_NAMES, default=DEFAULT_DATASET)


def _add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--split",
        choices=("chronological", "shuffled"),
        default=DEFAULT_SPLIT,
        help="how to hold rows out; 'shuffled' is train_test_split, as taught",
    )
    parser.add_argument(
        "--no-detrend",
        dest="detrend",
        action="store_false",
        default=DETREND,
        help="fit the tree on raw sales instead of on each store's distance from its "
        "own trend line; shows what the trend was worth",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockout",
        description="retail sales and product demand prediction",
    )
    parser.add_argument("--version", action="version", version=f"stockout {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_datasets = sub.add_parser("datasets", help="what --dataset accepts, and what is ready")
    p_datasets.set_defaults(handler=run_datasets)

    p_fetch = sub.add_parser("fetch", help="download the Rossmann archive from Kaggle")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if present")
    p_fetch.set_defaults(handler=run_fetch)

    p_describe = sub.add_parser("describe", help="rows, dtypes, nulls, demand balance")
    _add_data_args(p_describe)
    p_describe.set_defaults(handler=run_describe)

    p_models = sub.add_parser("models", help="what --regressor and --classifier accept")
    p_models.set_defaults(handler=run_models)

    p_compare = sub.add_parser("compare", help="every registered model on one split")
    _add_data_args(p_compare)
    p_compare.add_argument("--task", choices=TASK_NAMES, default="regression")
    _add_split_args(p_compare)
    p_compare.set_defaults(handler=run_compare)

    p_leakage = sub.add_parser(
        "leakage", help="what the shuffled split adds to each score, in points"
    )
    _add_data_args(p_leakage)
    p_leakage.add_argument("--task", choices=TASK_NAMES, default="regression")
    _add_split_args(p_leakage)
    p_leakage.set_defaults(handler=run_leakage)

    p_train = sub.add_parser("train", help="fit both models and save one artefact")
    _add_data_args(p_train)
    p_train.add_argument(
        "--regressor", choices=model_names("regression"), default=DEFAULT_REGRESSOR
    )
    p_train.add_argument(
        "--classifier", choices=model_names("classification"), default=DEFAULT_CLASSIFIER
    )
    _add_split_args(p_train)
    p_train.set_defaults(handler=run_train)

    p_predict = sub.add_parser("predict", help="sales and demand class for one store-day")
    p_predict.add_argument("--store", default=None, help="entity id; defaults to the first")
    p_predict.add_argument("--date", required=True, help="YYYY-MM-DD")
    p_predict.add_argument("--promo", action="store_true", help="a promotion runs that day")
    p_predict.add_argument("--state-holiday", default="0", choices=("0", "a", "b", "c"))
    p_predict.add_argument("--school-holiday", action="store_true")
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
