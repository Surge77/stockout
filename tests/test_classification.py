"""Scoring three ordered classes."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.config import DEMAND_CLASS_LABELS
from stockout.errors import BacktestError
from stockout.evaluate.classification import CLASS_CODES, confusion, report, score

PERFECT = pd.Series([0, 1, 2, 0, 1, 2])


def test_a_perfect_classifier_scores_one() -> None:
    scored = score(PERFECT, PERFECT)
    assert scored.accuracy == 1.0
    assert scored.macro_f1 == 1.0
    assert scored.adjacent_accuracy == 1.0


def test_macro_f1_punishes_dropping_the_smallest_class() -> None:
    """Accuracy would barely notice; macro-F1 loses a third of itself."""
    truth = pd.Series([0] * 8 + [1] * 8 + [2] * 2)
    always_low = pd.Series([0] * 18)
    scored = score(truth, always_low)
    assert scored.accuracy > 0.4
    assert scored.macro_f1 < 0.3
    assert scored.per_class_recall["High"] == 0.0


def test_adjacent_accuracy_forgives_a_one_class_miss_and_not_a_two() -> None:
    """Low predicted as High is a worse error than Low predicted as Medium, and F1
    charges the same for both."""
    truth = pd.Series([0, 0])
    near, far = pd.Series([1, 1]), pd.Series([2, 2])
    assert score(truth, near).adjacent_accuracy == 1.0
    assert score(truth, far).adjacent_accuracy == 0.0


def test_recall_is_reported_for_every_class_in_order() -> None:
    scored = score(PERFECT, PERFECT)
    assert list(scored.per_class_recall) == list(DEMAND_CLASS_LABELS)


def test_rows_without_a_label_are_dropped_in_pairs() -> None:
    """A closed day has no true class and gets no predicted one."""
    truth = pd.Series([0, 1, None, 2], dtype="Int64")
    predicted = pd.Series([0, 1, None, 2], dtype="Int64")
    assert score(truth, predicted).rows == 3


def test_scoring_nothing_says_so(  ) -> None:
    empty = pd.Series([], dtype="Int64")
    with pytest.raises(BacktestError, match="no labelled rows"):
        score(empty, empty)


def test_the_confusion_matrix_is_always_three_by_three() -> None:
    """sklearn infers the label set from the data, so a model that never predicts High
    would otherwise produce a silently smaller matrix."""
    truth, predicted = pd.Series([0, 0, 1]), pd.Series([0, 0, 0])
    matrix = confusion(truth, predicted)
    assert matrix.shape == (len(CLASS_CODES), len(CLASS_CODES))


def test_the_confusion_matrix_is_labelled_with_words_not_codes() -> None:
    matrix = confusion(PERFECT, PERFECT)
    assert list(matrix.columns) == list(DEMAND_CLASS_LABELS)
    assert matrix.index.name == "true"


def test_the_report_names_the_classes() -> None:
    text = report(PERFECT, PERFECT)
    assert all(label in text for label in DEMAND_CLASS_LABELS)
