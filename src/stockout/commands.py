"""What each subcommand does. The argument surface lives in `cli.py`.

Handlers return an exit code and never raise past `main`: a domain error becomes a
one-line `error: ...` on stderr, because a traceback is not a user interface.
"""

from __future__ import annotations

import argparse

from . import leakage, targets
from .config import MODEL_PATH, Task
from .data.adapters import SOURCES, available
from .datasets import describe, prepare, summary_line
from .evaluate import baseline_accuracy, compare, split, to_markdown
from .persistence import load as load_artifact
from .predict import predict
from .registry import specs_for
from .train import DEFAULT_CLASSIFIER, DEFAULT_REGRESSOR, train


def run_datasets(args: argparse.Namespace) -> int:
    for source in SOURCES:
        mark = "ready" if available(source) else "not downloaded"
        print(f"  {source.name:<12} {source.description}  [{mark}]")
        if source.raw_files:
            print(f"  {'':<12} needs {', '.join(source.raw_files)} — {source.origin}")
    return 0


def run_fetch(args: argparse.Namespace) -> int:
    from .data.download import fetch

    for path in fetch(force=args.force):
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
    entity = args.store or (artifact.entities[0] if artifact.entities else "")
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
