"""One file holding everything needed to answer a prediction request.

A fitted pipeline on its own cannot serve one. Four other things have to travel with it,
and each is easy to forget:

**Both models.** The product answers two questions per store-day — how much will it sell,
and is that Low, Medium or High for this shop — and they are different models that can
disagree. Shipping one would leave half the panel unable to answer.

**The demand thresholds.** A predicted number is a number. Turning it into a class needs
that store's own cut points, which are a property of the training window, not of the model.

**The label encoders.** Serving must use exactly the mapping training used. Refitting at
predict time would silently renumber the store types and answer confidently with nonsense.

**The store metadata.** Competition distance and promotion calendar are features, and a
single-row request carries only a store id and a date.

`joblib.load` executes code in the file it reads, so an artefact is exactly as trustworthy
as whoever wrote it. `load` refuses paths outside the artefact directory as a guard against
a careless caller, not as a sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import ARTIFACT_DIR, MODEL_PATH, Task
from .errors import StockoutError
from .targets import DemandThresholds
from .trend import StoreTrends

#: Bumped whenever the artefact's shape changes. An old file read by new code is a worse
#: failure than a refusal, because it usually half-works.
ARTIFACT_VERSION = 8


@dataclass
class Artifact:
    """Everything needed to answer a prediction request, in one file."""

    regressor: Any
    classifier: Any
    regressor_name: str
    classifier_name: str
    thresholds: DemandThresholds
    encoders: dict[str, Any]
    feature_columns: list[str]
    stores: pd.DataFrame | None
    dataset: str
    entity_label: str

    #: What `sales` counts, and whether `entity` is a store-item pair. Copied off the
    #: source at training time because serving holds an artefact and never the `Source`
    #: that produced it — and `has_products` decides how `features` reads an entity, so a
    #: default that disagreed with the fit would build a different matrix at serving.
    unit: str
    has_products: bool
    entities: list[str]
    scores: dict[str, dict[str, float]]
    n_train: int
    n_test: int

    #: A second regressor, fitted on every feature **except** `entity_code`, for
    #: answering about a store the fit has never seen. The deployed regressor cannot do
    #: that job: `entity_code` is one of its inputs, so a store it does not recognise has
    #: no valid value to supply. Two questions, two models.
    cold_regressor: Any = None
    cold_feature_columns: list[str] = field(default_factory=list)
    cold_scores: dict[str, float] = field(default_factory=dict)

    #: The entity ordering `entity_code` was built against, or `None` where the ids are
    #: numeric and are their own code. Without it a one-row serving frame re-derives the
    #: ordering from that one row, so every request answers about whichever series sorts
    #: first — see `features._entity_codes`. `entities` above is the same list when this
    #: is set; they are kept apart because one is a guard and this one is a mapping.
    entity_vocabulary: list[str] | None = None

    #: The span of the dataset this was built from — the dates for which a true figure
    #: exists. Deliberately the dataset's range and not the training split's: the question
    #: these answer is "is there an actual to compare against", which is a property of the
    #: data, not of the fold. They travel because serving needs them and cannot recover
    #: them, the artefact holding models and metadata but never the sales frame.
    #:
    #: `window.check` classifies a request against this pair, and `serving_row` caps the
    #: monotone store-age features at `observed_through` so a date past the end does not
    #: quietly feed the model a value beyond every split point it was built from.
    observed_from: pd.Timestamp | None = None
    observed_through: pd.Timestamp | None = None

    #: The per-store lines the regressor's target was measured against. Without these the
    #: saved model is unusable: it predicts a *residual*, and a residual with no line under
    #: it is not a sales figure. They are also the only part of the artefact that can
    #: answer a date past the data, the tree having no way to reach a level it never saw.
    trends: StoreTrends | None = None
    version: int = ARTIFACT_VERSION
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def model_for(self, task: Task) -> Any:
        return self.regressor if task == "regression" else self.classifier

    def name_for(self, task: Task) -> str:
        return self.regressor_name if task == "regression" else self.classifier_name

    def provenance(self) -> str:
        lines = [
            f"dataset: {self.dataset} ({len(self.entities):,} {self.entity_label}s)",
            f"regression:     {self.regressor_name}  {_scored(self.scores.get('regression'))}",
            f"classification: {self.classifier_name}  {_scored(self.scores.get('classification'))}",
            f"trained {self.trained_at} on {self.n_train:,} rows, held out {self.n_test:,}",
        ]
        return "\n".join(lines)


def _scored(scores: dict[str, float] | None) -> str:
    if not scores:
        return "(not scored)"
    # ASCII only: this is printed by notebook kernels too, and a Windows kernel's
    # cp1252 stdout replaces a middle dot with a literal question mark.
    return " | ".join(f"{name}={value:,.4f}" for name, value in scores.items())


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


def exists(path: Path = MODEL_PATH) -> bool:
    return path.exists()
