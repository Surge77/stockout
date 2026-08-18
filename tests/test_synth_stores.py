"""The generated metadata must contain every shape the preprocessor claims to handle."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.synth_stores import make_stores

#: The smallest fixture anything in this repo uses. Every assertion below runs at this
#: size, because a pattern that only appears in a large draw is not available to the
#: tests that need it.
SMALL = 4


def test_one_row_per_store() -> None:
    frame = make_stores(n_stores=6)
    assert len(frame) == 6
    assert list(frame[s.STORE]) == [1, 2, 3, 4, 5, 6]


def test_the_same_seed_produces_an_identical_frame() -> None:
    """Same contract as `synth.make_sales`, and `scripts/make_sample.py` relies on it."""
    assert make_stores(n_stores=8, seed=7).equals(make_stores(n_stores=8, seed=7))


def test_a_different_seed_produces_different_numbers() -> None:
    """Otherwise the seed is decoration and the determinism test above proves nothing."""
    first = make_stores(n_stores=8, seed=7)[s.COMPETITION_DISTANCE]
    second = make_stores(n_stores=8, seed=8)[s.COMPETITION_DISTANCE]
    assert not first.equals(second)


@pytest.mark.parametrize("n_stores", [0, -1])
def test_asking_for_no_stores_is_rejected(n_stores: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        make_stores(n_stores=n_stores)


def test_the_dtypes_match_the_schema() -> None:
    frame = make_stores(n_stores=SMALL)
    for column, dtype in s.STORE_DTYPES.items():
        assert frame[column].dtype == pd.api.types.pandas_dtype(dtype), column


# --- the shapes the ColumnTransformer branches need ------------------------------------


def test_every_store_type_appears_in_the_smallest_fixture() -> None:
    """OneHotEncoder is only exercised by a column that has more than one level."""
    assert set(make_stores(n_stores=SMALL)[s.STORE_TYPE]) == {"a", "b", "c", "d"}


def test_more_than_one_assortment_appears_in_the_smallest_fixture() -> None:
    assert make_stores(n_stores=SMALL)[s.ASSORTMENT].nunique() >= 2


def test_a_missing_competition_distance_appears_in_the_smallest_fixture() -> None:
    """The median-imputer branch is untested without one, and would report itself covered."""
    assert make_stores(n_stores=SMALL)[s.COMPETITION_DISTANCE].isna().any()


def test_a_missing_competition_date_appears_in_the_smallest_fixture() -> None:
    assert make_stores(n_stores=SMALL)[s.COMPETITION_OPEN_MONTH].isna().any()


def test_the_competition_month_and_year_are_missing_together_or_not_at_all() -> None:
    """Rossmann never ships one without the other, and code may rely on the pair."""
    frame = make_stores(n_stores=12)
    assert frame[s.COMPETITION_OPEN_MONTH].isna().equals(frame[s.COMPETITION_OPEN_YEAR].isna())


def test_competition_distance_is_positive_where_it_is_present() -> None:
    distances = make_stores(n_stores=12)[s.COMPETITION_DISTANCE].dropna()
    assert (distances > 0).all()


def test_a_promotion_calendar_appears_and_is_comma_separated_months() -> None:
    """TfidfVectorizer needs something with more than one token to vectorise."""
    intervals = make_stores(n_stores=SMALL)[s.PROMO_INTERVAL].dropna()
    assert not intervals.empty
    assert all(len(str(value).split(",")) == 4 for value in intervals)


# --- missingness that carries meaning --------------------------------------------------


def test_the_promo2_fields_are_absent_exactly_when_the_store_does_not_run_it() -> None:
    """The absence *is* the value. An imputer filling these invents a promotion.

    Asserted as an equivalence rather than as two one-way checks, so neither a store
    with dates and no flag nor a store with a flag and no dates can slip through.
    """
    frame = make_stores(n_stores=12)
    not_running = frame[s.PROMO2] == 0
    for column in (s.PROMO2_SINCE_WEEK, s.PROMO2_SINCE_YEAR, s.PROMO_INTERVAL):
        assert frame[column].isna().equals(not_running), column


def test_both_promotion_states_appear_in_the_smallest_fixture() -> None:
    assert set(make_stores(n_stores=SMALL)[s.PROMO2]) == {0, 1}


def test_the_promo2_start_week_is_a_real_week_number() -> None:
    weeks = make_stores(n_stores=12)[s.PROMO2_SINCE_WEEK].dropna()
    assert weeks.between(1, 52).all()
