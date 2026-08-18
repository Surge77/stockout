"""The one path from two CSVs to a model-ready frame."""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout import config
from stockout.data import schemas as s
from stockout.dataset import load_prepared, prepare, save_prepared
from stockout.errors import SchemaError
from stockout.features.store_features import IS_PROMO2_MONTH
from stockout.split.strategies import assert_date_major
from stockout.targets import DEMAND_CLASS


@pytest.fixture(scope="module")
def prepared():
    return prepare(horizon=7)


def test_the_committed_samples_prepare_without_arguments(prepared) -> None:
    """A fresh clone must be able to run everything offline."""
    assert prepared.rows > 0
    assert prepared.horizon == 7


def test_the_store_metadata_is_joined(prepared) -> None:
    assert s.STORE_TYPE in prepared.frame.columns
    assert IS_PROMO2_MONTH in prepared.frame.columns


def test_the_frame_comes_back_date_major(prepared) -> None:
    """Everything downstream that splits by position depends on this."""
    assert_date_major(prepared.frame)


def test_the_lag_warmup_rows_are_dropped_not_imputed(prepared) -> None:
    """Filling them would invent history the store does not have."""
    assert prepared.rows_dropped > 0
    assert prepared.frame[f"{s.SALES}_roll_mean_91"].notna().all()


def test_the_dropped_rows_are_counted_rather_than_silently_lost(prepared) -> None:
    assert prepared.rows + prepared.rows_dropped == prepared.rows_in
    assert f"{prepared.rows_dropped:,} dropped" in prepared.summary()


def test_labels_are_attached_by_default(prepared) -> None:
    assert DEMAND_CLASS in prepared.frame.columns
    assert prepared.thresholds.stores > 0


def test_labelling_can_be_skipped() -> None:
    """Anything that scores a model refits the thresholds on its own training slice."""
    plain = prepare(horizon=7, label=False)
    assert DEMAND_CLASS not in plain.frame.columns
    assert plain.thresholds.stores == 0


def test_a_missing_store_file_is_an_error_not_a_silent_skip(tmp_path: Path) -> None:
    """Carrying on without the join yields a frame that looks fine and has lost every
    categorical, text and competition feature. Every model still fits and nothing says why."""
    with pytest.raises(SchemaError, match="store metadata is missing"):
        prepare(stores_path=tmp_path / "absent.csv")


def test_a_missing_committed_sample_names_the_script_that_regenerates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "sample_stores.csv"
    monkeypatch.setattr(config, "SAMPLE_STORES_PATH", missing)
    with pytest.raises(SchemaError, match="make_sample"):
        prepare()


def test_a_prepared_frame_round_trips_through_parquet(prepared, tmp_path: Path) -> None:
    """CSV would re-parse dtypes and lose the categorical and nullable-integer columns."""
    path = save_prepared(prepared, tmp_path / "prepared.parquet")
    back = load_prepared(path, horizon=7)
    assert len(back) == prepared.rows
    assert back[DEMAND_CLASS].dtype == prepared.frame[DEMAND_CLASS].dtype


def test_loading_at_the_wrong_horizon_says_so(prepared, tmp_path: Path) -> None:
    """A cache prepared at horizon 7 has no lag_42, and a silent mismatch is worse."""
    path = save_prepared(prepared, tmp_path / "prepared.parquet")
    with pytest.raises(SchemaError, match="horizon 42"):
        load_prepared(path, horizon=42)
