"""Save a trained model to disk with the facts needed to interpret its predictions.

A fitted pipeline on its own cannot serve a request: the caller also needs the feature
column order, which dataset and task it was fitted for, and what it scored. All of it
travels in one joblib file.

**Nothing here is a trust boundary.** `joblib.load` executes code in the file it reads,
so an artefact is exactly as trustworthy as whoever wrote it. `load` refuses paths
outside the configured artefact directory as a guard against a careless caller, not as
a sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from .config import ARTIFACT_DIR, MODEL_PATH, Task
from .errors import StockoutError

#: Bumped whenever the artefact's shape changes. An old file loaded by new code is a
#: worse failure than a refusal, because it usually half-works.
ARTIFACT_VERSION = 2


@dataclass
class Artifact:
    """Everything needed to answer a prediction request, in one file."""

    model: Any
    dataset: str
    task: Task
    model_name: str
    features: list[str]
    target: str
    scores: dict[str, float]
    n_train: int
    n_test: int
    version: int = ARTIFACT_VERSION
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def provenance(self) -> str:
        # ASCII only: this string is printed by notebook kernels too, and a Windows
        # kernel's cp1252 stdout replaces a middle dot with a literal question mark.
        scored = " | ".join(f"{k}={v:,.4f}" for k, v in self.scores.items())
        return (
            f"{self.model_name} on {self.dataset} ({self.task} -> {self.target})\n"
            f"trained {self.trained_at} on {self.n_train:,} rows, "
            f"held out {self.n_test:,}\n{scored}"
        )


def save(artifact: Artifact, path: Path = MODEL_PATH) -> Path:
    """Write the artefact, creating the artefact directory if it is not there yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load(path: Path = MODEL_PATH) -> Artifact:
    """Read an artefact from inside the artefact directory, and nowhere else."""
    resolved = path.resolve()
    root = ARTIFACT_DIR.resolve()
    if not resolved.is_relative_to(root):
        raise StockoutError(f"refusing to load a model from outside {root}")
    if not resolved.exists():
        raise StockoutError(f"no model at {resolved}; run `stockout train` first")

    artifact = joblib.load(resolved)
    if not isinstance(artifact, Artifact):
        raise StockoutError(f"{resolved.name} does not hold a stockout model")
    if artifact.version != ARTIFACT_VERSION:
        raise StockoutError(
            f"{resolved.name} is version {artifact.version}, this build reads "
            f"{ARTIFACT_VERSION}; retrain it"
        )
    return artifact
