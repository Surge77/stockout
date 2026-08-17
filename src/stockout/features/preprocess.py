"""One `ColumnTransformer`, three branches, and the reason each column is in the one it is.

Every model in this package sits behind this. That is the point of a `Pipeline`: the
imputer's medians, the scaler's means and the encoder's vocabulary are all learned inside
a fold, from training rows only. Fit them outside and the test set has quietly voted on
its own preprocessing — the most common leak in tutorial code, and the one the leakage
notebook measures on purpose.

**Numeric** — `SimpleImputer(median)` then `StandardScaler`. Median rather than mean
because `competition_distance` is heavily right-skewed: a handful of stores are tens of
kilometres from anything, and a mean imputation would place every unknown store further
from a competitor than most real ones are. Scaling matters to Ridge, Lasso, KNN and SVM
and is irrelevant to trees; it is applied unconditionally because the cost is nil and a
per-model preprocessor would be four preprocessors to keep in step.

**Categorical** — `OneHotEncoder(handle_unknown="ignore")`. Four columns, of four, three,
four and seven levels. `handle_unknown` matters at serving time, where a store type the
training fold never saw must produce a row of zeros rather than an exception.

**Ordinal** — `store`, passed through as its integer id. Not one-hot encoded, and this is
the load-bearing decision in the module. 1115 levels would make the matrix sparse, which
`HistGradientBoosting` rejects outright and which `StandardScaler` would densify into
several gigabytes; it would make every KNN distance between two different stores the same
constant; and it would be 1115 coefficients nobody can defend. The store's *level* is
already carried causally by `sales_roll_mean_*`, which is learned from its own history
rather than from 1115 free parameters. ADR 0019.

**Text** — `TfidfVectorizer` on `promo_interval`. Honestly: three distinct values in the
whole dataset, so on its own this is a scaled one-hot in a costume, and it is constant per
store and constant in time. The feature that actually earns its place is
`store_features.is_promo2_month`, which crosses the same column with the row's calendar.
Both are here, and the pair is the honest answer to "what does the vectoriser buy you".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ..data import schemas as s

#: Month tokens are two to four letters, so the default pattern's two-character minimum
#: would be fine — but it is stated rather than inherited, because a silently changed
#: default here empties the vocabulary and the branch keeps running with zero features.
_TOKEN_PATTERN = r"(?u)\b[A-Za-z]{3,4}\b"


@dataclass(frozen=True)
class ColumnRoles:
    """Which encoding each column gets. Built from a frame, consumed by the transformer."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    ordinal: tuple[str, ...]
    text: str | None

    @property
    def all_columns(self) -> list[str]:
        text = [self.text] if self.text else []
        return [*self.numeric, *self.categorical, *self.ordinal, *text]

    def __post_init__(self) -> None:
        if not self.all_columns:
            raise ValueError("a preprocessor needs at least one column to transform")


def column_roles(frame: pd.DataFrame, *, feature_columns: list[str]) -> ColumnRoles:
    """Sort `feature_columns` into encoding roles by consulting the schema, then dtype.

    Schema first and dtype second, deliberately. `day_of_week` is stored as an int8 and
    is categorical; `store` is an int32 and is neither. Left to dtype alone, both would
    be scaled as though the distance from Monday to Thursday were three of something.
    """
    present = [column for column in feature_columns if column in frame.columns]

    categorical = tuple(c for c in present if c in s.CATEGORICAL_FEATURES)
    ordinal = tuple(c for c in present if c in s.ORDINAL_FEATURES)
    text = s.TEXT_FEATURE if s.TEXT_FEATURE in present else None

    spoken_for = {*categorical, *ordinal, *([text] if text else [])}
    numeric = tuple(
        c
        for c in present
        if c not in spoken_for and pd.api.types.is_numeric_dtype(frame[c])
    )
    return ColumnRoles(numeric=numeric, categorical=categorical, ordinal=ordinal, text=text)


def make_preprocessor(roles: ColumnRoles, *, drop_first: bool = False) -> ColumnTransformer:
    """The fitted-inside-a-fold preprocessor every model in this package shares.

    `drop_first` drops one level per categorical column. It belongs on the linear models,
    where keeping every level plus an intercept makes the design matrix singular — the
    dummy-variable trap. It is wrong for trees, which lose a usable split, and moot for
    Ridge and Lasso, whose penalty resolves the collinearity anyway.

    Output is dense. `sparse_threshold=0` forces it, because the Tf-idf branch is sparse
    by default and would otherwise drag the whole matrix sparse — which is exactly what
    ADR 0019 avoids by not one-hot encoding `store`.
    """
    branches: list[tuple[str, Pipeline | str, list[str] | str]] = []

    if roles.numeric:
        branches.append(("numeric", _numeric_branch(), list(roles.numeric)))
    if roles.categorical:
        branches.append(("categorical", _categorical_branch(drop_first), list(roles.categorical)))
    if roles.ordinal:
        branches.append(("ordinal", "passthrough", list(roles.ordinal)))
    if roles.text:
        # A bare string, not a one-element list. A list hands the branch a DataFrame and
        # TfidfVectorizer wants a 1-D iterable of documents; the error it raises does not
        # say so.
        branches.append(("text", _text_branch(), roles.text))

    return ColumnTransformer(
        transformers=branches,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _numeric_branch() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def _categorical_branch(drop_first: bool) -> Pipeline:
    return Pipeline(
        [
            # `schemas.STORE_DTYPES` stores these as pandas' nullable `string`, whose
            # missing marker is `pd.NA`. SimpleImputer detects gaps with `X != X`, and
            # `pd.NA != pd.NA` is not False — it is `pd.NA`, which raises
            # "boolean value of NA is ambiguous" from inside sklearn, several frames
            # below anything this package wrote. Normalise to object/np.nan first.
            ("as_labels", FunctionTransformer(_as_labels, feature_names_out="one-to-one")),
            # Categorical gaps are filled with a literal marker rather than the most
            # frequent level. "we do not know this store's type" is not evidence that it
            # is the commonest type, and the encoder can learn from the marker directly.
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            # `handle_unknown="ignore"` regardless of `drop_first`: a level the training
            # fold never saw becomes a row of zeros rather than an exception, which is
            # what a served request needs. sklearn warns when it happens, and that
            # warning is worth keeping — an unseen store type at serving time is a real
            # event, not noise.
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first" if drop_first else None,
                    sparse_output=False,
                ),
            ),
        ]
    )


def _text_branch() -> Pipeline:
    return Pipeline(
        [
            # TfidfVectorizer cannot accept a null and says so unhelpfully. An empty
            # document is the honest encoding: this store repeats its promotion in no
            # months, because it runs no continuing promotion.
            ("fill", FunctionTransformer(_as_documents, feature_names_out="one-to-one")),
            ("vectorise", TfidfVectorizer(token_pattern=_TOKEN_PATTERN, lowercase=True)),
        ]
    )


def _as_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Plain object columns whose gaps are `np.nan`, which is the only gap sklearn reads."""
    return frame.astype(object).where(frame.notna(), np.nan)


def _as_documents(column: pd.Series) -> pd.Series:
    """Null-free strings, which is all `TfidfVectorizer` will accept."""
    return column.fillna("").astype(str)
