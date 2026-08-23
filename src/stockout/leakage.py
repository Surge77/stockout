"""What the taught split is worth, measured rather than asserted.

Everywhere else in this package the honest choice is argued for in a docstring. Here it is
priced. `measure` fits the same model twice — once on a shuffled `train_test_split` and
once on a chronological hold-out — and reports the difference. That difference is the
optimism: the part of the score that comes from how the rows were divided rather than from
anything the model learned.

**Why it is not zero even without lag features.** Nothing in `features.py` is derived by
shifting the target, so the textbook leak cannot happen. What remains is subtler.
`entity_code` together with `month`, `day_of_month` and `day_of_week` very nearly picks
out a single row of the frame, so a tree with enough depth can isolate a test row inside a
leaf grown from the days immediately around it. In `data/synth.py` the residual left after
the calendar is an AR(1) with correlation 0.7 at one day, which is exactly what makes those
neighbours worth having. A shuffled split hands them over; a chronological one puts a wall
between the fitting period and the scoring period and does not.

That AR(1) is in the generator for this measurement and no other reason — see structure 5
in `data/synth.py`. With independent noise there is nothing for a neighbour to tell you,
both arms score the same, and the experiment cannot distinguish a good split from a bad
one. This module is what that design decision was for.

Two arms, not four. An earlier sketch of this file compared lagged and unlagged features as
well, which needs lag features to exist; they deliberately do not.

    python -m stockout leakage
    python -m stockout leakage --model random_forest --task regression
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import DEFAULT_SPLIT, RANDOM_SEED, TEST_SIZE, Task
from .datasets import Prepared, prepare
from .evaluate import evaluate, split
from .registry import model_names

#: The metric that carries the comparison for each task.
_HEADLINE: dict[str, str] = {"regression": "r2", "classification": "accuracy"}


@dataclass(frozen=True)
class Optimism:
    """One model, scored under both splits."""

    model: str
    task: Task
    metric: str
    shuffled: float
    chronological: float

    @property
    def difference(self) -> float:
        """How much the shuffled split adds. Positive means it flatters the model."""
        return self.shuffled - self.chronological

    @property
    def relative(self) -> float:
        """The difference as a share of the honest score, or 0 when that score is 0."""
        if not self.chronological:
            return 0.0
        return self.difference / abs(self.chronological)

    def line(self) -> str:
        return (
            f"{self.model:<18} {self.metric:<9} "
            f"shuffled {self.shuffled:>8.4f}   "
            f"chronological {self.chronological:>8.4f}   "
            f"optimism {self.difference:>+8.4f}"
        )


def measure(
    dataset: str | Prepared = "synthetic",
    task: Task = "regression",
    *,
    models: tuple[str, ...] | None = None,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> list[Optimism]:
    """Score each model under both splits and return the gap.

    Both arms share `prepared`, the same `test_size` and the same seed, so the split is the
    only thing that differs. Anything else varying would make the difference unattributable
    — which is the whole quantity being measured.
    """
    prepared = prepare(dataset) if isinstance(dataset, str) else dataset
    metric = _HEADLINE[task]

    wanted = models if models is not None else model_names(task)

    arms = {
        strategy: split(prepared, task, test_size=test_size, seed=seed, strategy=strategy)
        for strategy in ("shuffled", DEFAULT_SPLIT)
    }

    results: list[Optimism] = []
    for name in wanted:
        scored = {
            strategy: float(evaluate(name, task, data, seed=seed)[metric])  # type: ignore[arg-type]
            for strategy, data in arms.items()
        }
        results.append(
            Optimism(
                model=name,
                task=task,
                metric=metric,
                shuffled=scored["shuffled"],
                chronological=scored[DEFAULT_SPLIT],
            )
        )
    return results


def to_frame(results: list[Optimism]) -> pd.DataFrame:
    """The results as a table, for `to_markdown` or a notebook."""
    return pd.DataFrame(
        [
            {
                "model": item.model,
                "metric": item.metric,
                "shuffled": item.shuffled,
                "chronological": item.chronological,
                "optimism": item.difference,
            }
            for item in results
        ]
    )


def summary(results: list[Optimism]) -> str:
    """One line per model, plus the worst case — which is the number worth quoting."""
    if not results:
        return "no models measured"

    lines = [item.line() for item in results]
    worst = max(results, key=lambda item: item.difference)
    lines.append("")
    lines.append(
        f"largest optimism: {worst.model} at {worst.difference:+.4f} "
        f"({worst.relative:+.1%} of its honest score)"
    )
    return "\n".join(lines)
