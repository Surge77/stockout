"""What each subcommand does. The argument surface lives in `cli.py`.

Handlers return an exit code and never raise past `main`: a domain error becomes a
one-line `error: ...` on stderr, because a traceback is not a user interface.
"""

from __future__ import annotations

import argparse

from .config import MODEL_PATH, Task
from .datasets import dataset_for, describe, load
from .evaluate import compare, to_markdown
from .persistence import load as load_artifact
from .predict import describe_prediction, parse_assignments, predict, prompt_for
from .registry import model_names
from .train import train

#: What `train` fits when `--model` is not given. Lasso is the model the assignment
#: serves predictions from; logistic is the simplest thing that classifies.
DEFAULT_MODEL: dict[str, str] = {"regression": "lasso", "classification": "logistic"}


def run_describe(args: argparse.Namespace) -> int:
    dataset = dataset_for(args.dataset)
    frame = load(dataset)

    print(f"{dataset.name}: {len(frame):,} rows x {len(frame.columns)} columns")
    print(f"{dataset.description}")
    print(f"tasks: {', '.join(dataset.tasks())}\n")
    print(to_markdown(describe(frame, dataset).reset_index(names="column"), places=2))
    return 0


def run_compare(args: argparse.Namespace) -> int:
    dataset = dataset_for(args.dataset)
    task: Task = args.task
    results = compare(dataset, task, test_size=args.test_size, seed=args.seed)

    held_out = int(len(load(dataset)) * args.test_size)
    print(
        f"{dataset.name} · {task} -> {dataset.target_for(task)} · "
        f"{len(results)} models · {held_out:,} held-out rows\n"
    )
    print(to_markdown(results))
    print(f"\nbest: {_winner(results, task)}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    dataset = dataset_for(args.dataset)
    task: Task = args.task
    model_name = args.model or DEFAULT_MODEL[task]

    artifact, path = train(
        dataset, task, model_name, test_size=args.test_size, seed=args.seed
    )
    print(artifact.provenance())
    print(f"saved {path}")
    return 0


def run_predict(args: argparse.Namespace) -> int:
    artifact = load_artifact(MODEL_PATH)

    if args.interactive:
        values = prompt_for(artifact.features)
    else:
        values = parse_assignments(args.set or [])

    print(describe_prediction(artifact, predict(artifact, values)))
    return 0


def run_models(args: argparse.Namespace) -> int:
    for task in ("regression", "classification"):
        print(f"{task}: {', '.join(model_names(task))}")
    return 0


def _winner(results, task: Task) -> str:
    metric = "accuracy" if task == "classification" else "r2"
    best = results.loc[results[metric].idxmax()]
    return f"{best['model']} ({metric}={best[metric]:,.4f})"
