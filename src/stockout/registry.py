"""One list of models per task, so the CLI, the notebook and the results table agree.

Names repeat across tasks by design where the same algorithm serves both. The task is
the argument you already passed, so it does not also belong in the name.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED, Task
from .errors import StockoutError
from .models import CLASSIFIERS, REGRESSORS, ModelSpec

_BY_TASK: dict[str, tuple[ModelSpec, ...]] = {
    "regression": REGRESSORS,
    "classification": CLASSIFIERS,
}

TASK_NAMES: tuple[str, ...] = tuple(_BY_TASK)


def specs_for(task: Task) -> tuple[ModelSpec, ...]:
    """Every model registered for a task, simplest first."""
    if task not in _BY_TASK:
        raise StockoutError(f"unknown task {task!r}; choose from {', '.join(TASK_NAMES)}")
    return _BY_TASK[task]


def model_names(task: Task) -> tuple[str, ...]:
    """What `--model` accepts. Registry order, not alphabetical."""
    return tuple(spec.name for spec in specs_for(task))


def spec_for(name: str, *, task: Task) -> ModelSpec:
    for spec in specs_for(task):
        if spec.name == name:
            return spec
    raise StockoutError(
        f"unknown {task} model {name!r}; choose from {', '.join(model_names(task))}"
    )


def build(name: str, *, task: Task, seed: int = RANDOM_SEED) -> Pipeline:
    """An unfitted pipeline. A fresh object every call, never a cached one."""
    return spec_for(name, task=task).build(seed)
