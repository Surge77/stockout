"""Turn labels and ids into the integers a model can read, using mappings that travel.

Every mapping here is fixed before any data is read, or carried in the artefact, and both
amount to the same rule: **serving must encode exactly the way training did**. A second
`LabelEncoder` fitted at predict time would renumber the store types and answer
confidently with nonsense; an entity code re-derived from a one-row frame would answer
about whichever series sorted first.

Kept apart from `features.py` because it is the one part with no opinion about retail. It
knows about vocabularies and codes; what a store type *is* belongs next to the columns.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data import schemas as s

#: Anything outside a column's known vocabulary becomes this. Present in every
#: vocabulary below, so `LabelEncoder.transform` always has a code for it.
UNKNOWN = "unknown"

#: Fixed up front rather than learned. These are closed sets: Rossmann has four store
#: types, three assortments and four holiday codes, and all of them are known before any
#: data is read.
LABEL_VOCABULARIES: dict[str, tuple[str, ...]] = {
    s.STORE_TYPE: ("a", "b", "c", "d", UNKNOWN),
    s.ASSORTMENT: ("a", "b", "c", UNKNOWN),
    s.STATE_HOLIDAY: ("0", "a", "b", "c", UNKNOWN),
}

#: The code for an entity the vocabulary does not contain. Never produced in training, so
#: it lands outside every split point the trees were built from rather than impersonating
#: a real series. `predict.serving_row` refuses unknown entities before they reach here;
#: this is what shows up if that guard is ever bypassed.
UNKNOWN_ENTITY_CODE = -1

def encode_labels(
    frame: pd.DataFrame, *, encoders: dict[str, LabelEncoder] | None = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode every categorical against its fixed vocabulary.

    Returns the encoders so the artefact can carry them: serving must use exactly the
    mapping training used, and refitting at predict time would silently renumber.
    """
    frame = frame.copy()
    encoders = encoders or {}
    for column, vocabulary in LABEL_VOCABULARIES.items():
        if column not in frame.columns:
            continue
        if column not in encoders:
            encoder = LabelEncoder()
            encoder.fit(list(vocabulary))
            encoders[column] = encoder
        known = set(vocabulary)
        values = frame[column].astype("string").fillna(UNKNOWN)
        values = values.where(values.isin(known), UNKNOWN)
        frame[column] = encoders[column].transform(values.to_numpy()).astype("int16")
    return frame, encoders

def entity_codes(entities: pd.Series, vocabulary: list[str] | None = None) -> pd.Series:
    """Stable integer per entity, by whichever route the ids allow.

    **A carried vocabulary wins.** Position in a fixed list is the same number whether the
    frame holds one row or a million, which is the property serving needs and the only one
    that survives a single-row request.

    **Numeric ids are their own code.** Rossmann's entities are store numbers, so the id
    *is* a stable integer and no vocabulary is needed. Kept as the fast path deliberately:
    routing Rossmann through a vocabulary would renumber every store and move scores that
    are published.

    **Anything else falls back to `factorize`**, which is correct only when the frame is
    the whole training frame. A store-item entity such as `"3_26"` is not numeric, so
    before the vocabulary existed a one-row serving frame factorized it to 0 and the model
    answered about whichever series happened to sort first.
    """
    if vocabulary is not None:
        lookup = {name: index for index, name in enumerate(vocabulary)}
        codes = entities.astype("string").map(lookup)
        return codes.fillna(UNKNOWN_ENTITY_CODE).astype("int32")

    numeric = pd.to_numeric(entities, errors="coerce")
    if not numeric.isna().any():
        return numeric.astype("int32")
    return pd.Series(
        pd.factorize(entities.astype("string"), sort=True)[0], index=entities.index
    ).astype("int32")


def as_code(values: pd.Series) -> pd.Series:
    """An id column as an integer, with anything unreadable marked rather than guessed."""
    return (
        pd.to_numeric(values, errors="coerce")
        .fillna(UNKNOWN_ENTITY_CODE)
        .astype("int32")
    )
