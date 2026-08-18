"""The command line, exercised end to end against synthetic data and no network.

The commands that fit scikit-learn pipelines live in `test_cli_commands.py` — they carry
most of the runtime. What is left here is the cheap surface: the parser, the two data
commands, the baselines, and the errors that must arrive as a line rather than a
traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout.cli import main
from stockout.data.loaders import read_sales


def test_synth_writes_a_valid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "generated.csv"
    assert main(["synth", "--out", str(out), "--stores", "2", "--days", "120"]) == 0
    assert out.exists()
    assert len(read_sales(out)) > 0
    assert "wrote" in capsys.readouterr().out


def test_describe_reports_shape_and_gaps(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["describe", "--data", str(data_file)]) == 0
    out = capsys.readouterr().out
    assert "rows" in out
    assert "trading days" in out
    assert "calendar gaps" in out


def test_backtest_prints_a_table_and_a_verdict(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backtest", "--data", str(data_file), "--model", "seasonal_naive"]) == 0
    out = capsys.readouterr().out
    assert "rolling-origin folds" in out
    assert "MASE" in out
    assert "| fold" in out


def test_backtest_defaults_to_the_seasonal_naive_baseline(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backtest", "--data", str(data_file)]) == 0
    assert "seasonal_naive" in capsys.readouterr().out


def test_backtest_accepts_a_sliding_window(data_file: Path) -> None:
    assert main(["backtest", "--data", str(data_file), "--sliding", "--folds", "3"]) == 0


def test_an_unknown_model_is_rejected_by_the_parser(data_file: Path) -> None:
    """argparse `choices` means a typo fails before any data is read."""
    with pytest.raises(SystemExit) as excinfo:
        main(["backtest", "--data", str(data_file), "--model", "xgboost"])
    assert excinfo.value.code == 2


def test_a_missing_data_file_is_a_one_line_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["describe", "--data", str(tmp_path / "absent.csv")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_too_many_folds_is_a_one_line_error(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 730 days of fixture, 365 of which the minimum training window claims. At the
    # seven-day default horizon that leaves room for 52 folds, so 60 is the ask that
    # cannot be met.
    assert main(["backtest", "--data", str(data_file), "--folds", "60"]) == 1
    assert "cannot support" in capsys.readouterr().err


def test_no_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_fetch_reports_where_the_archive_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from stockout.data import download

    monkeypatch.setattr(download, "fetch", lambda force=False: tmp_path)
    assert main(["fetch"]) == 0
    assert "archive ready" in capsys.readouterr().out


def test_a_kaggle_failure_is_a_one_line_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The expected first-run failure must not arrive as a traceback."""
    from stockout.data import download
    from stockout.errors import DownloadError

    def refuse(force: bool = False) -> None:
        raise DownloadError("accept the competition rules first")

    monkeypatch.setattr(download, "fetch", refuse)
    assert main(["fetch"]) == 1
    assert "error: accept the competition rules first" in capsys.readouterr().err
