"""The ColumnTransformer: which column lands in which branch, and what comes out."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.features.preprocess import ColumnRoles, column_roles, make_preprocessor
from stockout.features.store_features import MONTH_TOKENS


def _frame() -> pd.DataFrame:
    """Four rows carrying one column of every role, including a missing value each."""
    return pd.DataFrame(
        {
            s.STORE: pd.array([1, 2, 3, 4], dtype="int32"),
            s.DAY_OF_WEEK: pd.array([1, 2, 3, 4], dtype="int8"),
            s.STORE_TYPE: pd.array(["a", "b", "a", None], dtype="string"),
            s.ASSORTMENT: pd.array(["a", "c", "a", "c"], dtype="string"),
            s.COMPETITION_DISTANCE: [100.0, 200.0, None, 400.0],
            s.PROMO_INTERVAL: pd.array(
                ["Jan,Apr,Jul,Oct", "", "Mar,Jun,Sept,Dec", ""], dtype="string"
            ),
        }
    )


def _roles(frame: pd.DataFrame) -> ColumnRoles:
    return column_roles(frame, feature_columns=list(frame.columns))


def _dense(matrix: object) -> np.ndarray:
    """`fit_transform` is typed as returning anything, including None. It does not."""
    return np.asarray(matrix, dtype="float64")


# --- sorting columns into roles ----------------------------------------------------------


def test_the_schema_decides_the_role_before_the_dtype_does() -> None:
    """`day_of_week` is an int8 and is categorical; `store` is an int32 and is neither.

    Left to dtype alone both would be scaled, as though Monday to Thursday were three
    of something.
    """
    roles = _roles(_frame())
    assert s.DAY_OF_WEEK in roles.categorical
    assert s.STORE in roles.ordinal
    assert s.DAY_OF_WEEK not in roles.numeric
    assert s.STORE not in roles.numeric


def test_the_store_id_is_ordinal_and_never_one_hot() -> None:
    """ADR 0019. 1115 levels would force the matrix sparse and exclude HistGB."""
    assert _roles(_frame()).ordinal == (s.STORE,)


def test_the_promotion_calendar_is_the_text_column() -> None:
    assert _roles(_frame()).text == s.PROMO_INTERVAL


def test_a_plain_float_column_is_numeric() -> None:
    assert s.COMPETITION_DISTANCE in _roles(_frame()).numeric


def test_columns_absent_from_the_frame_are_ignored() -> None:
    """A caller's feature list may name a column this particular frame does not carry."""
    roles = column_roles(_frame(), feature_columns=[s.STORE, "not_a_column"])
    assert roles.all_columns == [s.STORE]


def test_a_preprocessor_with_nothing_to_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        ColumnRoles(numeric=(), categorical=(), ordinal=(), text=None)


# --- what the transformer produces -------------------------------------------------------


def test_the_output_is_dense_because_the_text_branch_would_otherwise_win() -> None:
    """Tf-idf is sparse by default; HistGradientBoosting refuses sparse input."""
    out = make_preprocessor(_roles(_frame())).fit_transform(_frame())
    assert isinstance(out, np.ndarray)
    assert not hasattr(out, "toarray")


def test_no_missing_value_survives_the_preprocessor() -> None:
    """Both gaps in the fixture — a numeric one and a categorical one — must be filled."""
    out = _dense(make_preprocessor(_roles(_frame())).fit_transform(_frame()))
    assert not np.isnan(out).any()


def test_the_numeric_branch_scales_to_zero_mean() -> None:
    frame = _frame()
    roles = ColumnRoles(numeric=(s.COMPETITION_DISTANCE,), categorical=(), ordinal=(), text=None)
    out = make_preprocessor(roles).fit_transform(frame)
    assert out.mean() == pytest.approx(0.0, abs=1e-9)


def test_a_missing_number_is_imputed_with_the_median_not_the_mean() -> None:
    """`competition_distance` is right-skewed; a mean would place every unknown store
    further from a competitor than most real ones are."""
    frame = pd.DataFrame({s.COMPETITION_DISTANCE: [1.0, 2.0, 3.0, 1000.0, None]})
    roles = ColumnRoles(numeric=(s.COMPETITION_DISTANCE,), categorical=(), ordinal=(), text=None)
    pipeline = make_preprocessor(roles).fit(frame)
    imputer = pipeline.named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(2.5)


def test_the_store_id_passes_through_unscaled() -> None:
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(s.STORE,), text=None)
    out = make_preprocessor(roles).fit_transform(_frame())
    assert list(out[:, 0]) == [1, 2, 3, 4]


def test_an_unknown_category_at_predict_time_becomes_zeros_rather_than_an_error() -> None:
    """A store type the training fold never saw must not take down a served request."""
    train = _frame()
    unseen = train.copy()
    unseen[s.STORE_TYPE] = pd.array(["z", "z", "z", "z"], dtype="string")

    roles = ColumnRoles(numeric=(), categorical=(s.STORE_TYPE,), ordinal=(), text=None)
    pipeline = make_preprocessor(roles).fit(train)
    assert pipeline.transform(unseen).sum() == 0.0


def test_a_missing_category_is_marked_unknown_rather_than_filled_with_the_commonest() -> None:
    """Not knowing a store's type is not evidence that it is the commonest type."""
    roles = ColumnRoles(numeric=(), categorical=(s.STORE_TYPE,), ordinal=(), text=None)
    pipeline = make_preprocessor(roles).fit(_frame())
    assert any("unknown" in str(name) for name in pipeline.get_feature_names_out())


# --- the text branch ---------------------------------------------------------------------


def test_the_text_branch_produces_one_feature_per_month_token() -> None:
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(), text=s.PROMO_INTERVAL)
    pipeline = make_preprocessor(roles).fit(_frame())
    names = set(pipeline.get_feature_names_out())
    assert {"jan", "apr", "jul", "oct", "mar", "jun", "dec"} <= names


def test_rossmanns_four_letter_september_survives_tokenisation() -> None:
    """The default token pattern would keep it; this asserts the one in use does too."""
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(), text=s.PROMO_INTERVAL)
    pipeline = make_preprocessor(roles).fit(_frame())
    assert "sept" in set(pipeline.get_feature_names_out())


def test_a_store_with_no_promotion_calendar_vectorises_to_zeros() -> None:
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(), text=s.PROMO_INTERVAL)
    out = _dense(make_preprocessor(roles).fit_transform(_frame()))
    assert out[1].sum() == 0.0


def test_a_fold_where_nobody_promotes_still_produces_every_month_column() -> None:
    """The vocabulary is fixed, so it cannot depend on which stores landed in the fold.

    Learned from the documents, this raised `empty vocabulary; perhaps the documents only
    contain stop words` four frames inside a `ColumnTransformer` — which is what fitting
    on a single store that runs no continuing promotion looks like.
    """
    nobody = pd.DataFrame({s.PROMO_INTERVAL: pd.array(["", "", ""], dtype="string")})
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(), text=s.PROMO_INTERVAL)
    pipeline = make_preprocessor(roles).fit(nobody)

    names = set(pipeline.get_feature_names_out())
    assert {"jan", "sept", "dec"} <= names
    assert _dense(pipeline.transform(nobody)).sum() == 0.0


def test_the_month_vocabulary_is_the_one_the_promotion_feature_reads() -> None:
    """Two consumers of the same tokens, so neither can drift on how September is spelt."""
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(), text=s.PROMO_INTERVAL)
    pipeline = make_preprocessor(roles).fit(_frame())
    assert set(pipeline.get_feature_names_out()) == set(MONTH_TOKENS)


def test_a_null_calendar_does_not_reach_the_vectoriser() -> None:
    """TfidfVectorizer rejects nulls, and says so unhelpfully when it does."""
    frame = pd.DataFrame({s.PROMO_INTERVAL: pd.array(["Jan,Apr", None], dtype="string")})
    roles = ColumnRoles(numeric=(), categorical=(), ordinal=(), text=s.PROMO_INTERVAL)
    assert _dense(make_preprocessor(roles).fit_transform(frame)).shape[0] == 2


# --- the dummy trap ------------------------------------------------------------------------


def test_dropping_the_first_level_removes_one_column_per_categorical() -> None:
    """Keeping every level plus an intercept makes a linear design matrix singular."""
    roles = _roles(_frame())
    kept = _dense(make_preprocessor(roles, drop_first=False).fit_transform(_frame()))
    dropped = _dense(make_preprocessor(roles, drop_first=True).fit_transform(_frame()))
    assert dropped.shape[1] == kept.shape[1] - len(roles.categorical)
