"""Scoring three ordered classes, and why accuracy alone will not do it.

**Accuracy has a floor and it is not zero.** Three classes fitted as terciles means a
model that guesses the commonest one is right about a third of the time. Reporting 71%
without the 33% floor beside it says nothing, which is why `DummyClassifier` is the first
row of every table in this project.

**Accuracy also hides which class failed.** The test window's classes drift — measured at
39/36/25 on the synthetic sample against 33/33/33 in training — so a model can drop the
smallest class entirely, score respectably, and be useless at exactly the job it was
built for. Macro-F1 averages the per-class scores without weighting by class size, so
losing the smallest class costs a full third of the metric. That is the number to lead
with; accuracy is reported next to it because everyone expects it.

**The classes are ordered, and most classification metrics do not know that.** Predicting
High when the truth is Low is a worse mistake than predicting Medium, and F1 charges the
same for both. `adjacent_accuracy` is here for that: the share of predictions that are
right or one class out. It is not a standard metric and is labelled as such — it exists
because a demand planner cares about the difference and macro-F1 cannot express it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from ..config import DEMAND_CLASS_LABELS
from ..errors import BacktestError

#: 0, 1, 2 in Low/Medium/High order. Passed explicitly to every sklearn call, because
#: their defaults infer the label set from whatever appeared in the data — so a model
#: that never predicts High produces a two-column confusion matrix and a report that
#: silently changes shape.
CLASS_CODES: tuple[int, ...] = tuple(range(len(DEMAND_CLASS_LABELS)))


@dataclass(frozen=True)
class ClassificationScores:
    """What one classifier did, on the rows that carried a label."""

    rows: int
    accuracy: float
    macro_f1: float
    adjacent_accuracy: float
    per_class_recall: dict[str, float]

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "test_rows": self.rows,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "adjacent": self.adjacent_accuracy,
        }
        for label, value in self.per_class_recall.items():
            row[f"recall_{label.lower()}"] = value
        return row


def score(y_true: pd.Series, y_predicted: pd.Series) -> ClassificationScores:
    """Score a classifier on the rows where both a truth and a prediction exist.

    Rows are dropped in pairs. A closed day has no true class and gets no predicted one,
    so it appears in neither; scoring it as a miss would penalise every model equally for
    a question none of them was asked.
    """
    truth, predicted = _aligned(y_true, y_predicted)
    if truth.size == 0:
        raise BacktestError("no labelled rows to score")

    return ClassificationScores(
        rows=int(truth.size),
        accuracy=float(accuracy_score(truth, predicted)),
        # `zero_division` because the label set is forced to all three classes: if one
        # is absent from both the truth and the prediction its F1 is genuinely
        # undefined, and sklearn warns — which `filterwarnings = ["error"]` turns into
        # a crash. Scoring it 0 is the conservative reading: a class the model never
        # got right, for any reason, did not earn credit.
        macro_f1=float(
            f1_score(
                truth,
                predicted,
                labels=CLASS_CODES,
                average="macro",
                zero_division=cast(str, 0),
            )
        ),
        adjacent_accuracy=float(np.mean(np.abs(truth - predicted) <= 1)),
        per_class_recall=_recalls(truth, predicted),
    )


def confusion(y_true: pd.Series, y_predicted: pd.Series) -> pd.DataFrame:
    """Rows are truth, columns are prediction, both in Low/Medium/High order.

    Labelled with words rather than codes. A matrix indexed 0/1/2 requires the reader to
    remember which way round the encoding went, and that is exactly the thing that gets
    remembered wrongly.
    """
    truth, predicted = _aligned(y_true, y_predicted)
    matrix = confusion_matrix(truth, predicted, labels=CLASS_CODES)
    names = list(DEMAND_CLASS_LABELS)
    return pd.DataFrame(matrix, index=pd.Index(names, name="true"), columns=names)


def report(y_true: pd.Series, y_predicted: pd.Series) -> str:
    """sklearn's per-class table, with the class names attached."""
    truth, predicted = _aligned(y_true, y_predicted)
    return str(
        classification_report(
            truth,
            predicted,
            labels=CLASS_CODES,
            target_names=list(DEMAND_CLASS_LABELS),
            # A class the model never predicted has an undefined precision. Report it
            # as 0 rather than letting sklearn warn and emit NaN — a dropped class is
            # a real score of zero, not a missing measurement. Its stub types this as
            # str while the runtime accepts 0, hence the cast.
            zero_division=cast(str, 0),
        )
    )


def _recalls(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Share of each true class the model actually found.

    Recall rather than precision, because the failure this project cares about is a busy
    day predicted as quiet — a missed High is an empty shelf, and recall is the metric
    that counts those.
    """
    scores: dict[str, float] = {}
    for code, label in zip(CLASS_CODES, DEMAND_CLASS_LABELS, strict=True):
        actual = truth == code
        scores[label] = float(np.mean(predicted[actual] == code)) if actual.any() else float("nan")
    return scores


def _aligned(y_true: pd.Series, y_predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Both series as int arrays, over the rows where neither is missing."""
    frame = pd.DataFrame({"truth": y_true, "predicted": y_predicted}).dropna()
    return (
        frame["truth"].to_numpy(dtype="int64"),
        frame["predicted"].to_numpy(dtype="int64"),
    )
