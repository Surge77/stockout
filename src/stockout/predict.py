"""Serve one row from a saved model, either from flags or from `input()` prompts.

The interactive mode is step 5 of `Task_Regression.pdf`: prompt for each feature, build
a one-row frame, print the prediction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from .errors import StockoutError
from .persistence import Artifact


def parse_assignments(assignments: Sequence[str]) -> dict[str, float]:
    """Turn `["age=29", "internships=1"]` into a dict, rejecting anything malformed."""
    values: dict[str, float] = {}
    for assignment in assignments:
        name, separator, raw = assignment.partition("=")
        if not separator or not name.strip():
            raise StockoutError(f"expected name=value, got {assignment!r}")
        try:
            values[name.strip()] = float(raw)
        except ValueError as exc:
            raise StockoutError(f"{name.strip()} needs a number, got {raw!r}") from exc
    return values


def prompt_for(
    features: Sequence[str], *, reader: Callable[[str], str] = input
) -> dict[str, float]:
    """Ask for every feature in turn. Re-asks on a value that is not a number."""
    values: dict[str, float] = {}
    for feature in features:
        while True:
            raw = reader(f"Enter Your {feature}: ")
            try:
                values[feature] = float(raw)
                break
            except ValueError:
                print(f"  {raw!r} is not a number, try again")
    return values


def to_frame(values: dict[str, float], features: Sequence[str]) -> pd.DataFrame:
    """One row, columns in the order the model was fitted on."""
    missing = [feature for feature in features if feature not in values]
    if missing:
        raise StockoutError(f"missing values for: {', '.join(missing)}")

    unknown = sorted(set(values) - set(features))
    if unknown:
        raise StockoutError(f"not features of this model: {', '.join(unknown)}")

    return pd.DataFrame([[values[feature] for feature in features]], columns=list(features))


def predict(artifact: Artifact, values: dict[str, float]) -> float:
    """The model's answer for one row."""
    prediction = artifact.model.predict(to_frame(values, artifact.features))
    return float(prediction[0])


def describe_prediction(artifact: Artifact, prediction: float) -> str:
    """Phrase the answer for the task it came from."""
    label = artifact.target.replace("_", " ").title()
    if artifact.task == "classification":
        return f"Predicted {label}: {round(prediction)}"
    return f"Predicted {label}: {prediction:,.2f}"
