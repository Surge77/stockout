"""One list of models per task, so the CLI, the notebooks and the results table agree.

Three consumers need to know what models exist, and if each kept its own list they would
drift — the notebook would chart five and the table would print six, and nobody would
notice which one went missing. They all read from here.

Names repeat across tasks on purpose: `random_forest` is a regressor *and* a classifier,
and calling the second one `random_forest_classifier` would put the task in the name of
every model in half the registry. The task is the argument you already passed.
"""

from __future__ import annotations

from ..config import RANDOM_SEED
from ..errors import BacktestError
from .adapter import SklearnClassifier, SklearnForecaster
from .classifiers import BASELINE_CLASSIFIERS, BASELINE_REGRESSORS, LINEAR_CLASSIFIERS
from .kernels import KERNEL_CLASSIFIERS, KERNEL_REGRESSORS
from .linear import LINEAR_REGRESSORS
from .spec import ModelSpec, Task
from .trees import TREE_CLASSIFIERS, TREE_REGRESSORS

#: Ordered from simplest to most complex, and the results table keeps this order. A table
#: sorted by score answers "which won"; a table in this order also answers "did the extra
#: complexity pay", which is the more useful question and the harder one to fake.
REGRESSORS: tuple[ModelSpec, ...] = (
    *BASELINE_REGRESSORS,
    *LINEAR_REGRESSORS,
    *TREE_REGRESSORS,
    *KERNEL_REGRESSORS,
)

CLASSIFIERS: tuple[ModelSpec, ...] = (
    *BASELINE_CLASSIFIERS,
    *LINEAR_CLASSIFIERS,
    *TREE_CLASSIFIERS,
    *KERNEL_CLASSIFIERS,
)

_BY_TASK: dict[str, tuple[ModelSpec, ...]] = {
    "regression": REGRESSORS,
    "classification": CLASSIFIERS,
}


def specs_for(task: Task) -> tuple[ModelSpec, ...]:
    """Every model registered for a task, simplest first."""
    if task not in _BY_TASK:
        raise BacktestError(f"unknown task {task!r}; choose from {sorted(_BY_TASK)}")
    return _BY_TASK[task]


def model_names(task: Task) -> tuple[str, ...]:
    """What `--model` accepts. Registry order, not alphabetical — see the module note."""
    return tuple(spec.name for spec in specs_for(task))


def spec_for(name: str, *, task: Task) -> ModelSpec:
    for spec in specs_for(task):
        if spec.name == name:
            return spec
    raise BacktestError(
        f"unknown {task} model {name!r}; choose from {', '.join(model_names(task))}"
    )


def build(
    name: str, *, task: Task, horizon: int = 0, seed: int = RANDOM_SEED
) -> SklearnForecaster | SklearnClassifier:
    """An unfitted model, wrapped so `backtest` can drive it.

    A fresh object every call, never a cached one. `evaluate/backtest.py` builds one per
    fold, and reusing a fitted estimator across folds leaks the previous fold's fit into
    the next — quietly, because the scores only improve a little.
    """
    spec = spec_for(name, task=task)
    wrapper = SklearnForecaster if task == "regression" else SklearnClassifier
    return wrapper(spec, horizon=horizon, seed=seed)
