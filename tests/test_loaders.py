"""Reading CSVs, including the dtype landmine Rossmann actually ships."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stockout.data import schemas as s
from stockout.data.loaders import canonicalise, read_sales, write_sales
from stockout.data.synth import make_sales


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
