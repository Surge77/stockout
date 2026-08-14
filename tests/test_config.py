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


def test_env_float_reads_a_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCKOUT_UNDERAGE_COST", "7.5")
    assert config.underage_cost() == 7.5


def test_env_float_warns_and_falls_back_on_a_malformed_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("STOCKOUT_OVERAGE_COST", "cheap")
    with caplog.at_level("WARNING"):
        assert config.overage_cost() == config.DEFAULT_OVERAGE_COST
    assert "STOCKOUT_OVERAGE_COST" in caplog.text


@pytest.mark.parametrize(
    "accessor, default",
    [
        ("gap_days", config.DEFAULT_GAP_DAYS),
        ("lead_time_days", config.DEFAULT_LEAD_TIME_DAYS),
        ("review_period_days", config.DEFAULT_REVIEW_PERIOD_DAYS),
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


def test_load_env_is_a_no_op_when_no_files_exist(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config.load_env()
