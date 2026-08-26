"""The integer the model reads as *which series is this*, and why it has to travel.

`entity_code` is a feature. Training builds it from the whole frame; serving builds it
from a single row through the same `features.build`. Nothing forces those two to agree,
and where they disagree the model is asked about one series and answers about another —
silently, with a plausible number.

For Rossmann they agree by luck: its entities are numeric strings, so the code is the id
itself and a one-row frame produces the same value a million-row frame does. A store-item
entity such as `"3_26"` is not numeric, falls to `pd.factorize`, and factorize numbers
whatever it is handed — position 1 of three entities in training, position 0 of one at
serving.

These tests pin both halves: the numeric path keeps its literal value, so Rossmann's
fitted codes do not move, and a carried vocabulary makes the non-numeric path agree
across frame sizes.
"""

from __future__ import annotations

import pandas as pd

from stockout import features
from stockout.data import schemas as s

DATE = pd.Timestamp("2014-06-02")


def _sales(entities: list[str]) -> pd.DataFrame:
    """One row per entity, carrying only what `features.build` requires."""
    return pd.DataFrame(
        [
            {
                s.ENTITY: entity,
                s.DATE: DATE,
                s.SALES: 100.0,
                s.OPEN: 1,
                s.PROMO: 0,
                s.STATE_HOLIDAY: "0",
                s.SCHOOL_HOLIDAY: 0,
            }
            for entity in entities
        ]
    )


def _code_for(frame: pd.DataFrame, entity: str, vocabulary: list[str] | None) -> int:
    built, _ = features.build(frame, None, entity_vocabulary=vocabulary)
    match = built[built[s.ENTITY].astype("string") == entity]
    return int(match[features.ENTITY_CODE].iloc[0])


def test_numeric_entities_keep_their_own_id() -> None:
    """Rossmann's path. The code is the store number, so nothing renumbers on a retrain."""
    frame = _sales(["1", "7", "1115"])
    assert _code_for(frame, "1115", None) == 1115


def test_a_single_row_matches_the_training_frame_for_a_named_entity() -> None:
    """The serving bug. Without a vocabulary the one-row frame factorizes to 0."""
    vocabulary = ["3_25", "3_26", "4_25"]
    training = _sales(vocabulary)
    serving = _sales(["3_26"])

    assert _code_for(serving, "3_26", vocabulary) == _code_for(training, "3_26", vocabulary)


def test_every_named_entity_gets_a_distinct_code() -> None:
    vocabulary = ["3_25", "3_26", "4_25"]
    built, _ = features.build(_sales(vocabulary), None, entity_vocabulary=vocabulary)
    assert built[features.ENTITY_CODE].nunique() == len(vocabulary)


def test_an_entity_outside_the_vocabulary_is_marked_rather_than_guessed() -> None:
    """`-1` never occurs in training, so a bypassed guard shows up instead of hiding.

    `predict.serving_row` refuses an unknown entity before it reaches here. If that check
    is ever removed, a plausible-looking code would answer confidently about the wrong
    series; a value no fitted split point covers is the louder failure.
    """
    built, _ = features.build(
        _sales(["9_99"]), None, entity_vocabulary=["3_25", "3_26"]
    )
    assert int(built[features.ENTITY_CODE].iloc[0]) == -1
