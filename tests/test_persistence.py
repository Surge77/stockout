"""What has to travel with a model, and the one path `load` refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout import config, persistence
from stockout.errors import SchemaError
from stockout.persistence import ARTIFACT_VERSION, Artifact, load, new_artifact, save
from stockout.targets import DemandThresholds


def _artifact(**overrides: object) -> Artifact:
    base: dict[str, object] = {
        "regressor": "a-fitted-pipeline",
        "classifier": "another-one",
        "thresholds": DemandThresholds(per_store={1: (10.0, 20.0)}, fallback=(5.0, 15.0)),
        "horizon": 7,
        "feature_columns": ["promo", "store"],
        "training_rows": 100,
        "scores": {"r2": 0.84},
    }
    return new_artifact(**{**base, **overrides})  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def artifacts_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never write a real model during a unit test."""
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)


def test_an_artifact_round_trips(tmp_path: Path) -> None:
    saved = save(_artifact())
    back = load(saved)
    assert back.horizon == 7
    assert back.feature_columns == ["promo", "store"]


def test_the_thresholds_travel_with_the_model() -> None:
    """Without them a served prediction is a number with no class beside it."""
    back = load(save(_artifact()))
    assert back.thresholds.for_store(1) == (10.0, 20.0)
    assert back.thresholds.for_store(999) == (5.0, 15.0)


def test_the_horizon_travels_with_the_model() -> None:
    """A model fitted at horizon 7 reads sales_lag_7 and cannot be told otherwise."""
    assert load(save(_artifact(horizon=42))).horizon == 42


def test_provenance_is_recorded() -> None:
    artifact = _artifact()
    assert "UTC" in artifact.trained_at
    assert "100" in artifact.summary()
    assert "r2 0.840" in artifact.summary()


def test_saving_creates_the_directory(tmp_path: Path) -> None:
    saved = save(_artifact(), tmp_path / "nested" / "deeper" / "model.joblib")
    assert saved.exists()


def test_a_missing_model_names_the_command_that_makes_one(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="stockout train"):
        load(tmp_path / "absent.joblib")


def test_loading_from_outside_the_artifact_directory_is_refused(tmp_path: Path) -> None:
    """`joblib.load` runs code from the file, so the path is not a free parameter."""
    outside = tmp_path.parent / "elsewhere.joblib"
    outside.write_bytes(b"not a model")
    with pytest.raises(SchemaError, match="refusing to load"):
        load(outside)


def test_a_file_that_is_not_a_model_is_refused(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "other.joblib"
    joblib.dump({"not": "an artifact"}, path)
    with pytest.raises(SchemaError, match="does not contain"):
        load(path)


def test_an_artifact_from_an_older_build_is_refused_rather_than_half_read(
    tmp_path: Path,
) -> None:
    import joblib

    stale = _artifact()
    stale.version = ARTIFACT_VERSION - 1
    path = tmp_path / "stale.joblib"
    joblib.dump(stale, path)
    with pytest.raises(SchemaError, match="artifact version"):
        load(path)


def test_the_default_path_sits_inside_the_artifact_directory(tmp_path: Path) -> None:
    assert persistence.artifact_path().parent == tmp_path
