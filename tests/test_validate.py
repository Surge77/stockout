"""Schema assertions — each one is a bug that has cost somebody an afternoon."""

from __future__ import annotations

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.synth import make_sales
from stockout.data.validate import calendar_gaps, null_profile, validate_sales
from stockout.errors import SchemaError


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_sales(n_stores=2, days=90, seed=5)


def test_a_clean_frame_passes(frame: pd.DataFrame) -> None:
    validate_sales(frame)


def test_missing_columns_are_named_in_the_error(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="promo"):
        validate_sales(frame.drop(columns=[s.PROMO]))


def test_a_string_date_column_is_rejected(frame: pd.DataFrame) -> None:
    stringly = frame.assign(**{s.DATE: frame[s.DATE].astype(str)})
    with pytest.raises(SchemaError, match="datetime64"):
        validate_sales(stringly)


def test_duplicate_store_date_rows_are_rejected(frame: pd.DataFrame) -> None:
    doubled = pd.concat([frame, frame.head(1)], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_sales(doubled)


def test_a_closed_store_with_sales_is_rejected(frame: pd.DataFrame) -> None:
    """The invariant that justifies excluding closed days from every metric."""
    broken = frame.copy()
    closed = broken.index[broken[s.OPEN] == 0][0]
    broken.loc[closed, s.SALES] = 1234.0
    with pytest.raises(SchemaError, match="open=0 but non-zero sales"):
        validate_sales(broken)


def test_a_non_binary_open_flag_is_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[broken.index[0], s.OPEN] = 2
    with pytest.raises(SchemaError, match="must be 0 or 1"):
        validate_sales(broken)


def test_negative_sales_are_rejected(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    trading = broken.index[broken[s.OPEN] == 1][0]
    broken.loc[trading, s.SALES] = -1.0
    with pytest.raises(SchemaError, match="parsing bug"):
        validate_sales(broken)


def test_null_profile_lists_every_column_worst_first(frame: pd.DataFrame) -> None:
    holed = frame.copy()
    holed.loc[holed.index[:3], s.CUSTOMERS] = None
    profile = null_profile(holed)
    assert len(profile) == len(frame.columns)
    assert profile.iloc[0]["column"] == s.CUSTOMERS
    assert profile.iloc[0]["nulls"] == 3


def test_calendar_gaps_is_empty_for_contiguous_data() -> None:
    contiguous = make_sales(n_stores=1, days=120)
    assert calendar_gaps(contiguous).empty
