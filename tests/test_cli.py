"""The command line, exercised end to end against synthetic data and no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout.cli import main
from stockout.data.loaders import read_sales, write_sales
from stockout.data.synth import make_sales


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    return write_sales(make_sales(n_stores=2, days=730, seed=13), tmp_path / "sales.csv")


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


def test_backtest_runs_the_gradient_boosted_model_and_reports_beating_the_baseline(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The model is reachable from the command line, not only from a notebook."""
    assert main(["backtest", "--data", str(data_file), "--model", "gbm", "--folds", "2"]) == 0
    out = capsys.readouterr().out
    assert "`gbm`" in out
    assert "beats seasonal-naive" in out


def test_frontier_prices_every_service_level_and_names_the_cheapest(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["frontier", "--data", str(data_file)]) == 0
    out = capsys.readouterr().out
    assert "newsvendor target quantile 0.75" in out
    assert "quantile crossing on" in out
    assert "| quantile" in out
    assert "Cheapest at quantile" in out


def test_frontier_costs_more_stock_for_more_service(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The curve has to slope, or the table is decoration."""
    assert main(["frontier", "--data", str(data_file)]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| 0.")]
    held = [float(row.split("|")[8].strip().replace(",", "")) for row in rows]
    assert held == sorted(held)
    assert held[0] < held[-1]


def test_frontier_without_lightgbm_says_what_to_install(
    data_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bare install must reach the model and then explain itself, not fail at import."""
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", None)
    assert main(["frontier", "--data", str(data_file)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: LightGBM is not installed")
    assert "[gbm]" in err


def test_frontier_rejects_a_store_that_is_not_in_the_window(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["frontier", "--data", str(data_file), "--store", "999"]) == 1
    assert "store 999 has no rows" in capsys.readouterr().err


def test_a_missing_data_file_is_a_one_line_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["describe", "--data", str(tmp_path / "absent.csv")]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_too_many_folds_is_a_one_line_error(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["backtest", "--data", str(data_file), "--folds", "40"]) == 1
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
