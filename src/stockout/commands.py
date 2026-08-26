"""What each subcommand does. The argument surface lives in `cli.py`.

Handlers return an exit code and never raise past `main`: a domain error becomes a
one-line `error: ...` on stderr, because a traceback is not a user interface.
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import leakage, ranking, targets
from .config import MODEL_PATH, Task
from .data import schemas as s
from .data.adapters import SOURCES, available
from .datasets import describe, prepare, summary_line
from .errors import StockoutError
from .evaluate import baseline_accuracy, compare, split, to_markdown
from .persistence import load as load_artifact
from .predict import predict
from .registry import specs_for
from .train import DEFAULT_CLASSIFIER, DEFAULT_REGRESSOR, train


def run_datasets(args: argparse.Namespace) -> int:
    for source in SOURCES:
        mark = "ready" if available(source) else "not downloaded"
        print(f"  {source.name:<19} {source.description}  [{mark}]")
        if source.raw_files:
            print(f"  {'':<19} needs {', '.join(source.raw_files)} — {source.origin}")
    return 0


def run_fetch(args: argparse.Namespace) -> int:
    from .data.download import fetch

    for path in fetch(args.dataset, force=args.force):
        print(f"  {path}")
    return 0


def run_describe(args: argparse.Namespace) -> int:
    prepared = prepare(args.dataset)
    print(summary_line(prepared))
    print(f"{prepared.source.description}\n")

    labels = targets.apply(prepared.frame, targets.fit(prepared.frame))
    balance = targets.class_balance(labels)
    shares = " | ".join(f"{name} {share:.1%}" for name, share in balance.items())
    print(f"demand classes (whole frame, for orientation only): {shares}\n")
    print(to_markdown(describe(prepared).reset_index(names="column"), places=2))
    return 0


def run_models(args: argparse.Namespace) -> int:
    for task in ("regression", "classification"):
        print(f"{task}:")
        for spec in specs_for(task):
            mark = "" if spec.taught else "   (not in the course material)"
            print(f"  {spec.name:<18} {spec.note}{mark}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    prepared = prepare(args.dataset)
    task: Task = args.task
    results = compare(
        prepared, task, test_size=args.test_size, seed=args.seed,
        strategy=args.split, detrend=args.detrend,
    )
    data = split(
        prepared, task, test_size=args.test_size, seed=args.seed,
        strategy=args.split, detrend=args.detrend,
    )

    print(f"{summary_line(prepared)}")
    trend_note = "per-store trend removed" if args.detrend else "raw sales, no trend term"
    print(
        f"{task} | {len(results)} models | {len(data.X_test):,} held-out rows | {trend_note}"
    )
    if task == "classification":
        print(f"floor: {baseline_accuracy(data.Y_test):.1%} (always guess the commonest class)")
    print()
    print(to_markdown(results))
    print(f"\nbest: {_winner(results, task)}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    artifact, path = train(
        args.dataset,
        regressor=args.regressor,
        classifier=args.classifier,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(artifact.provenance())
    print(f"saved {path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    artifact = load_artifact(MODEL_PATH)
    entity = _entity_for(artifact, store=args.store, item=args.item)
    answer = predict(
        artifact,
        entity=entity,
        date=args.date,
        promo=args.promo,
        state_holiday=args.state_holiday,
        school_holiday=args.school_holiday,
    )
    print("\n".join(answer.lines()))
    return 0


def _entity_for(artifact, *, store: str | None, item: str | None) -> str:
    """The series to ask about, from whichever ids the caller supplied.

    A store-day artefact takes a store and ignores `--item`; a product artefact needs both
    and says so rather than answering about an arbitrary item. Defaulting either half
    would return a confident number for a question nobody asked.
    """
    if not artifact.has_products:
        if item is not None:
            raise StockoutError(
                f"{artifact.dataset} records one row per {artifact.entity_label}-day; "
                f"there is no item axis to select"
            )
        return store or (artifact.entities[0] if artifact.entities else "")

    if store is None or item is None:
        raise StockoutError(
            f"{artifact.dataset} is a product dataset; give both --store and --item "
            f"(for example --store 3 --item 7)"
        )
    return f"{store}{s.PRODUCT_ENTITY_SEPARATOR}{item}"


def run_products(args: argparse.Namespace) -> int:
    """Every item one shop stocks on one day, busiest first.

    The list, not a single series, is what a person ordering stock actually needs — and it
    is scored in one pass rather than one request per item.
    """
    artifact = load_artifact(MODEL_PATH)
    store = args.store or (ranking.stores_in(artifact)[0] if artifact.has_products else "")
    shelf = ranking.rank(
        artifact,
        store=store,
        date=args.date,
        promo=args.promo,
        state_holiday=args.state_holiday,
        school_holiday=args.school_holiday,
    )

    print(f"store {shelf.store} on {shelf.date:%Y-%m-%d} | {len(shelf)} items")
    caveat = shelf.status.caveat() if shelf.status else None
    if caveat:
        print(f"forecast: {caveat}")
    print(f"predicted total {shelf.total:,.0f} {artifact.unit}")
    print()

    rows = [
        {
            "item": item.item,
            "predicted": round(item.sales, 1),
            "demand": item.demand_class,
            "quiet below": round(item.thresholds[0], 1),
            "busy above": round(item.thresholds[1], 1),
        }
        for item in shelf.busiest(args.top)
    ]
    print(to_markdown(pd.DataFrame(rows), places=1))
    return 0


def _winner(results, task: Task) -> str:  # type: ignore[no-untyped-def]
    metric = "accuracy" if task == "classification" else "r2"
    best = results.loc[results[metric].idxmax()]
    return f"{best['model']} ({metric}={best[metric]:,.4f})"


__all__ = [
    "DEFAULT_CLASSIFIER",
    "DEFAULT_REGRESSOR",
    "run_compare",
    "run_datasets",
    "run_describe",
    "run_fetch",
    "run_models",
    "run_predict",
    "run_products",
    "run_train",
]


def run_leakage(args: argparse.Namespace) -> int:
    """Score every model under both splits and print the gap between them.

    The number to read is the last line. It is the largest amount any model gained purely
    from being handed calendar-adjacent rows, and it is the correction that has to be
    applied in your head to any figure produced with `--split shuffled`.
    """
    prepared = prepare(args.dataset)
    task: Task = args.task
    results = leakage.measure(prepared, task, test_size=args.test_size, seed=args.seed)

    print(summary_line(prepared))
    print(f"{task} | {len(results)} models | shuffled vs chronological, same rows and seed")
    print()
    print(leakage.summary(results))
    return 0
