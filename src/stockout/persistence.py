"""Saving a trained system to disk, and what has to travel with the model.

A fitted pipeline on its own cannot serve a request. Three other things are needed and
each is easy to forget:

**The demand thresholds.** A predicted number is a number; turning it into Low, Medium or
High requires that store's tercile cut points, and those are a property of the *training
window*, not of the model. Ship the model without them and the classification half of the
product simply cannot answer.

**The horizon.** A model fitted at horizon 7 reads `sales_lag_7`. Handing it a frame
prepared at horizon 42 produces a `KeyError` at best and a silently wrong answer at worst,
because the column names differ but the shapes may not.

**Provenance.** When it was trained, on how many rows, and what it scored. A served
prediction with no way to ask "which model said this" is not something to put a login in
front of.

`joblib` rather than `pickle` because it stores the large numpy arrays inside a fitted
forest efficiently, and rather than ONNX or PMML because those would add a dependency and
a conversion step to solve a portability problem this project does not have.

**Nothing here is a trust boundary.** `joblib.load` executes code in the file it reads, so
an artifact is exactly as trustworthy as whoever wrote it. `load` refuses paths outside
the configured artifact directory for that reason — the web app takes an upload of *data*
from an admin, never of a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from . import config
from .errors import SchemaError
from .targets import DemandThresholds

#: Bumped whenever the artifact's shape changes. An old file loaded by new code is a
#: worse failure than a refusal, because it usually half-works.
ARTIFACT_VERSION = 1

DEFAULT_ARTIFACT_NAME = "model.joblib"


@dataclass
class Artifact:
    """Everything needed to answer a prediction request, in one file."""

    regressor: Any
    classifier: Any
    thresholds: DemandThresholds
    horizon: int
    feature_columns: list[str]
    trained_at: str
    training_rows: int
    scores: dict[str, float] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION

    def summary(self) -> str:
        scored = ", ".join(f"{name} {value:.3f}" for name, value in sorted(self.scores.items()))
        return (
            f"trained {self.trained_at} on {self.training_rows:,} rows "
            f"at horizon {self.horizon}; {len(self.feature_columns)} features"
            + (f"; {scored}" if scored else "")
        )


def new_artifact(
    *,
    regressor: Any,
    classifier: Any,
    thresholds: DemandThresholds,
    horizon: int,
    feature_columns: list[str],
    training_rows: int,
    scores: dict[str, float] | None = None,
) -> Artifact:
    """Stamp an artifact with the time it was made. UTC, so two machines agree."""
    return Artifact(
        regressor=regressor,
        classifier=classifier,
        thresholds=thresholds,
        horizon=horizon,
        feature_columns=list(feature_columns),
        trained_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        training_rows=training_rows,
        scores=dict(scores or {}),
    )


def artifact_path(name: str = DEFAULT_ARTIFACT_NAME) -> Path:
    """Where artifacts live. One directory, so `load` has something to check against."""
    return config.ARTIFACT_DIR / name


def save(artifact: Artifact, path: Path | str | None = None) -> Path:
    """Write an artifact, creating the directory if needed."""
    destination = Path(path) if path is not None else artifact_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return destination


def load(path: Path | str | None = None) -> Artifact:
    """Read an artifact back, refusing anything from outside the artifact directory.

    `joblib.load` executes code in the file it reads. That is fine for a file this
    package wrote and unacceptable for an arbitrary path a request supplied, so the
    check is here rather than left to each caller to remember.
    """
    source = Path(path) if path is not None else artifact_path()
    _require_inside_artifact_dir(source)

    if not source.exists():
        raise SchemaError(f"no model at {source}. Train one with `python -m stockout train`.")

    artifact = joblib.load(source)
    if not isinstance(artifact, Artifact):
        raise SchemaError(f"{source} does not contain a stockout model")
    if artifact.version != ARTIFACT_VERSION:
        raise SchemaError(
            f"{source} is artifact version {artifact.version}, this build reads "
            f"{ARTIFACT_VERSION}. Retrain rather than loading it."
        )
    return artifact


def _require_inside_artifact_dir(path: Path) -> None:
    root = config.ARTIFACT_DIR.resolve()
    try:
        resolved = path.resolve()
    except OSError as exc:  # pragma: no cover - platform-specific path failures
        raise SchemaError(f"cannot resolve {path}") from exc

    if not resolved.is_relative_to(root):
        raise SchemaError(
            f"refusing to load a model from outside {root}. Loading an artifact runs "
            "code from it, so the path is not something a caller gets to choose freely."
        )
