"""Reading CSVs, including the dtype landmine Rossmann actually ships."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.loaders import (
    canonicalise,
    canonicalise_store,
    merge_store,
    read_sales,
    read_store,
    write_sales,
)
from stockout.data.synth import make_sales
from stockout.errors import SchemaError


def test_rossmann_column_names_are_renamed_to_snake_case() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2015-01-01"],
            "Store": [1],
            "DayOfWeek": [4],
            "Sales": [5263],
            "Customers": [555],
            "Open": [1],
            "Promo": [0],
            "StateHoliday": ["0"],
            "SchoolHoliday": [1],
        }
    )
    out = canonicalise(raw)
    assert set(s.REQUIRED_COLUMNS).issubset(out.columns)
    assert "StateHoliday" not in out.columns


def test_state_holiday_unifies_integer_and_string_zeros() -> None:
    """Rossmann's train.csv genuinely mixes 0 and "0" in this column.

    Left alone, pandas infers `object`, and every `== "0"` comparison silently misses
    half the rows.
    """
    mixed = pd.DataFrame({"StateHoliday": [0, "0", "a"]})
    out = canonicalise(mixed)
    assert out[s.STATE_HOLIDAY].tolist() == ["0", "0", "a"]
    assert out[s.STATE_HOLIDAY].dtype == "string"


def test_dates_are_parsed_and_rows_sorted_by_store_then_date() -> None:
    shuffled = make_sales(n_stores=2, days=30).sample(frac=1.0, random_state=1)
    out = canonicalise(shuffled)
    assert pd.api.types.is_datetime64_any_dtype(out[s.DATE])
    assert out.equals(out.sort_values([s.STORE, s.DATE]).reset_index(drop=True))


def test_declared_dtypes_are_applied() -> None:
    out = canonicalise(make_sales(n_stores=1, days=10))
    for column, dtype in s.DTYPES.items():
        assert out[column].dtype == dtype


def test_a_write_read_round_trip_preserves_the_frame(tmp_path: Path) -> None:
    original = make_sales(n_stores=2, days=60, seed=2)
    path = write_sales(original, tmp_path / "nested" / "sales.csv")
    assert path.exists()
    pd.testing.assert_frame_equal(read_sales(path), original)


def test_reading_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_sales(tmp_path / "absent.csv")


def test_canonicalise_does_not_mutate_its_input() -> None:
    original = make_sales(n_stores=1, days=10)
    before = original.copy()
    canonicalise(original)
    pd.testing.assert_frame_equal(original, before)


# --- store.csv ------------------------------------------------------------------------


def _raw_store() -> pd.DataFrame:
    """Two stores in Rossmann's own spelling, including the NaNs it really ships."""
    return pd.DataFrame(
        {
            "Store": [2, 1],
            "StoreType": ["a", "c"],
            "Assortment": ["c", "a"],
            "CompetitionDistance": [570.0, None],
            "CompetitionOpenSinceMonth": [11.0, 9.0],
            "CompetitionOpenSinceYear": [2007.0, 2008.0],
            "Promo2": [1, 0],
            "Promo2SinceWeek": [13.0, None],
            "Promo2SinceYear": [2010.0, None],
            "PromoInterval": ["Jan,Apr,Jul,Oct", None],
        }
    )


def test_store_columns_are_renamed_to_snake_case() -> None:
    out = canonicalise_store(_raw_store())
    assert s.STORE_TYPE in out.columns
    assert s.COMPETITION_DISTANCE in out.columns
    assert s.PROMO_INTERVAL in out.columns
    assert "StoreType" not in out.columns


def test_store_rows_are_sorted_by_store_so_two_reads_align() -> None:
    out = canonicalise_store(_raw_store())
    assert list(out[s.STORE]) == [1, 2]


def test_a_missing_competition_distance_survives_the_reader_as_a_null() -> None:
    """Imputing here would decide, silently, what an absent competitor means."""
    out = canonicalise_store(_raw_store())
    assert out.loc[out[s.STORE] == 1, s.COMPETITION_DISTANCE].isna().all()


def test_reading_a_store_csv_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "store.csv"
    _raw_store().to_csv(path, index=False)
    assert list(read_store(path)[s.STORE]) == [1, 2]


def test_merging_keeps_every_sales_row(sales: pd.DataFrame) -> None:
    store = canonicalise_store(_raw_store())
    merged = merge_store(sales, store)
    assert len(merged) == len(sales)


def test_merging_attaches_the_metadata_to_the_right_store(sales: pd.DataFrame) -> None:
    store = canonicalise_store(_raw_store())
    merged = merge_store(sales, store)
    for store_id, expected in ((1, "c"), (2, "a")):
        rows = merged.loc[merged[s.STORE] == store_id, s.STORE_TYPE]
        assert (rows == expected).all()


def test_a_store_with_no_metadata_becomes_nulls_rather_than_vanishing(
    sales: pd.DataFrame,
) -> None:
    """A left join, deliberately. A dropped row is a data problem you cannot see."""
    store = canonicalise_store(_raw_store())
    merged = merge_store(sales, store)
    assert len(merged) == len(sales)
    assert merged.loc[merged[s.STORE] == 3, s.STORE_TYPE].isna().all()


def test_an_absent_promo_interval_becomes_an_empty_string_not_a_null(
    sales: pd.DataFrame,
) -> None:
    """TfidfVectorizer cannot vectorise a null, and "no months" is the honest encoding."""
    merged = merge_store(sales, canonicalise_store(_raw_store()))
    assert not merged[s.PROMO_INTERVAL].isna().any()
    assert (merged.loc[merged[s.STORE] == 1, s.PROMO_INTERVAL] == "").all()


def test_store_metadata_without_its_key_is_rejected(sales: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="store"):
        merge_store(sales, pd.DataFrame({s.STORE_TYPE: ["a"]}))
