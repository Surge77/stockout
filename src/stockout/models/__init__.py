"""The model registry: what `--model` on the command line is allowed to say.

Two families live here and they answer different questions.

**Baselines** (`baselines.py`) are per-store rules with no fitting worth the name —
same-weekday-last-week, last value, trailing mean. They exist to give every other
number something to be compared against. A gradient-boosted R² of 0.88 means nothing
until you know what a rule a shopkeeper could apply in their head scores.

**Pipelines** (`linear.py`, `trees.py`, `kernels.py`, `classifiers.py`) are
scikit-learn estimators wrapped in a `Pipeline` behind the shared preprocessor. They
are registered by name so the CLI, the notebooks and the comparison table all read
from one list rather than three that drift apart.

The registry maps a name to a *factory*, never to an instance. `evaluate/backtest.py`
builds a fresh model per fold, and reusing a fitted object across folds leaks the
previous fold's fit into the next one — quietly, because the scores only improve a
little.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .baselines import BASELINES

#: Every name the CLI accepts. Sorted so `--help` does not reorder between runs.
FORECASTER_NAMES: tuple[str, ...] = tuple(sorted(BASELINES))


def forecaster(name: str, *, horizon: int) -> Callable[[], Any]:
    """A zero-argument factory for `name`.

    `horizon` is accepted for every model and used by those whose features depend on
    it. The baselines ignore it: same-weekday-last-week is the same rule at a
    seven-day horizon as at a forty-two-day one, so there is nothing for them to do
    with the number.
    """
    if name in BASELINES:
        return BASELINES[name]
    raise KeyError(f"unknown forecaster {name!r}; choose from {', '.join(FORECASTER_NAMES)}")
