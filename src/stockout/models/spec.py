"""What the registry stores about a model: how to build it, and what it costs to run.

A `ModelSpec` is deliberately not an estimator. It is a *factory plus the facts a reader
needs to interpret the score* — which rows the model saw, whether its categoricals were
dummy-trapped, and why it is capped if it is. Those facts belong next to the number in
the comparison table, not in a footnote somebody has to go and find.

`build` takes the seed rather than closing over one, so a spec is a constant that can be
declared at module level and reused across folds without carrying state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

Task = Literal["regression", "classification"]


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the registry."""

    name: str
    task: Task
    build: Callable[[int], Any]
    """Given a random seed, return an unfitted estimator. Called once per fold."""

    drop_first: bool = False
    """Drop one level per one-hot column.

    True for the plain linear models, where keeping every level alongside an intercept
    makes the design matrix singular — the dummy-variable trap. False for trees, which
    lose a usable split by it, and for the penalised linear models, whose penalty
    resolves the collinearity anyway.
    """

    sample_rows: int | None = None
    """Cap on training rows, or None for all of them.

    Set only where the estimator's cost makes the full set impossible rather than merely
    slow. Reported alongside the score, because a model fitted on 5,000 rows and one
    fitted on 800,000 are not comparable and the table must not imply they are.
    """

    note: str = ""
    """One line on why this model is in the comparison. Printed in the results table."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a model spec needs a name")
        if self.sample_rows is not None and self.sample_rows < 1:
            raise ValueError(f"{self.name}: sample_rows must be positive when set")
