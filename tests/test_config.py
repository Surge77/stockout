"""Env knobs degrade to their defaults rather than killing a run mid-backtest."""

from __future__ import annotations

import pytest

from stockout import config


def test_env_int_reads_a_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKOUT_N_FOLDS", "9")
    assert config.n_folds() == 9


def test_env_int_falls_back_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STOCKOUT_N_FOLDS", raising=False)
    assert config.n_folds() == config.DEFAULT_N_FOLDS


def test_env_int_falls_back_on_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKOUT_HORIZON_DAYS", "   ")
    assert config.horizon_days() == config.DEFAULT_HORIZON_DAYS


def test_env_int_warns_and_falls_back_on_a_malformed_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("STOCKOUT_HORIZON_DAYS", "42d")
    with caplog.at_level("WARNING"):
        assert config.horizon_days() == config.DEFAULT_HORIZON_DAYS
    assert "STOCKOUT_HORIZON_DAYS" in caplog.text


def test_the_subsample_size_is_readable_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STOCKOUT_SUBSAMPLE_ROWS", "1500")
    assert config.subsample_rows() == 1500


@pytest.mark.parametrize(
    "accessor, default",
    [
        ("gap_days", config.DEFAULT_GAP_DAYS),
        ("n_folds", config.DEFAULT_N_FOLDS),
        ("horizon_days", config.DEFAULT_HORIZON_DAYS),
        ("subsample_rows", config.SUBSAMPLE_ROWS),
        ("random_seed", config.RANDOM_SEED),
    ],
)
def test_every_knob_has_a_working_default(
    accessor: str, default: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(f"STOCKOUT_{accessor.upper()}", raising=False)
    assert getattr(config, accessor)() == default


def test_the_horizon_default_is_long_enough_to_forbid_a_lag_of_one() -> None:
    """A one-day horizon lets yesterday's sales in, and the project loses its point."""
    assert config.DEFAULT_HORIZON_DAYS > 1


def test_the_demand_classes_are_ordered_low_to_high() -> None:
    """Not alphabetical, deliberately.

    `LabelEncoder` sorts its classes, which would map High->0 and Low->1. Every
    confusion-matrix caption and per-class recall would then be wrong while looking
    entirely plausible. This tuple is the order the rest of the package encodes
    against, so it is worth an assertion of its own.
    """
    assert config.DEMAND_CLASS_LABELS == ("Low", "Medium", "High")


def test_load_env_is_a_no_op_when_no_files_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config.load_env()
