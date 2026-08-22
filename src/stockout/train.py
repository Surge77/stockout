"""Fit one model on the training split and save it with its held-out scores."""

from __future__ import annotations

from pathlib import Path

from .config import MODEL_PATH, RANDOM_SEED, TEST_SIZE, Task
from .datasets import Dataset
from .evaluate import score, split
from .persistence import Artifact, save
from .registry import build


def train(
    dataset: Dataset,
    task: Task,
    model_name: str,
    *,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
    path: Path = MODEL_PATH,
) -> tuple[Artifact, Path]:
    """Fit, score on the held-out rows, and write the artefact."""
    X_train, X_test, Y_train, Y_test = split(dataset, task, test_size=test_size, seed=seed)

    model = build(model_name, task=task, seed=seed)
    model.fit(X_train, Y_train)

    artifact = Artifact(
        model=model,
        dataset=dataset.name,
        task=task,
        model_name=model_name,
        features=list(dataset.features),
        target=dataset.target_for(task),
        scores=score(task, Y_test, model.predict(X_test)),
        n_train=len(X_train),
        n_test=len(X_test),
    )
    return artifact, save(artifact, path)
