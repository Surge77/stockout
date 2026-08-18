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

**The two families are one list to the command line and two lists in here.** A baseline
reads three columns and needs no preprocessing; a pipeline needs the whole prepared
frame. `forecaster` hides that difference from `--model` and nowhere else, because a
user choosing between `seasonal_naive` and `ridge` is choosing between two forecasts,
not between two internal representations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .baselines import BASELINES
from .registry import build, model_names

#: Baselines first and alphabetical, then the registry in its own order — simplest
#: model first. Stable across runs either way, which is what `--help` needs; the order
#: also reads as an argument, since a reader scanning it meets the floor before the
#: things that have to clear it.
BASELINE_NAMES: tuple[str, ...] = tuple(sorted(BASELINES))
REGRESSOR_NAMES: tuple[str, ...] = model_names("regression")
FORECASTER_NAMES: tuple[str, ...] = BASELINE_NAMES + REGRESSOR_NAMES


def forecaster(name: str, *, horizon: int) -> Callable[[], Any]:
    """A zero-argument factory for `name`, baseline or scikit-learn pipeline alike.

    `horizon` is accepted for every model and used by those whose features depend on
    it. The baselines ignore it: same-weekday-last-week is the same rule at a
    seven-day horizon as at a forty-two-day one, so there is nothing for them to do
    with the number.

    A factory rather than a model, because `evaluate/backtest.py` needs a fresh one per
    fold — see the module docstring.
    """
    if name in BASELINES:
        return BASELINES[name]
    if name in REGRESSOR_NAMES:
        return lambda: build(name, task="regression", horizon=horizon)
    raise KeyError(f"unknown forecaster {name!r}; choose from {', '.join(FORECASTER_NAMES)}")
