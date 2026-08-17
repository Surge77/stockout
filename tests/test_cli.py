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


def _column(out: str, name: str) -> list[float]:
    """Read one named column out of a printed markdown table.

    By header rather than by position. The first version of this counted pipes, and when
    a cost column was inserted in the middle it went on passing while asserting a
    different column than the one it named — a positional index into a table that grows
    is a test that lies rather than one that fails.
    """
    lines = out.splitlines()
    header = next(line for line in lines if line.startswith("| quantile"))
    index = [cell.strip() for cell in header.split("|")].index(name)
    return [
        float(line.split("|")[index].strip().replace(",", ""))
        for line in lines
        if line.startswith("| 0.")
    ]


def test_frontier_costs_more_stock_for_more_service(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The curve has to slope, or the table is decoration."""
    assert main(["frontier", "--data", str(data_file)]) == 0
    held = _column(capsys.readouterr().out, "mean_on_hand")
    assert held == sorted(held)
    assert held[0] < held[-1]


def test_frontier_charges_nothing_for_a_pipeline_it_never_opened(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR 0011 is on by default, and by default there is no lorry for it to bill."""
    assert main(["frontier", "--data", str(data_file)]) == 0
    assert _column(capsys.readouterr().out, "transit_cost") == pytest.approx([0.0] * 6)


def test_frontier_bills_the_pipeline_and_can_be_told_not_to(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two halves of ADR 0011: charged by default, waivable for a supplier-owned pipeline."""
    argv = ["frontier", "--data", str(data_file), "--lead-time", "7"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "transit charged at 1.0/unit/day" in out
    assert all(cost > 0.0 for cost in _column(out, "transit_cost"))

    assert main([*argv, "--transit-holding-cost", "0"]) == 0
    free = capsys.readouterr().out
    assert _column(free, "transit_cost") == pytest.approx([0.0] * 6)


def test_frontier_rejects_a_negative_transit_rate_before_it_trains_anything(
    data_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", None)
    argv = ["frontier", "--data", str(data_file), "--transit-holding-cost", "-1"]
    assert main(argv) == 1
    assert "error: --transit-holding-cost cannot be negative" in capsys.readouterr().err


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


def test_frontier_rejects_a_review_period_before_it_trains_anything(
    data_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rejecting it after the fit would waste seconds and arrive as a traceback.

    LightGBM is removed to prove the order: if the check ran late, this would fail with
    the missing-dependency message instead.
    """
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", None)
    assert main(["frontier", "--data", str(data_file), "--review-period", "0"]) == 1
    assert "error: --review-period must be at least 1 day" in capsys.readouterr().err


def test_frontier_rejects_a_store_that_is_not_in_the_window(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["frontier", "--data", str(data_file), "--store", "999"]) == 1
    assert "store 999 has no rows" in capsys.readouterr().err


def test_frontier_names_the_system_it_priced_so_two_runs_cannot_be_confused(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lead time changes what was simulated, so it has to change the header too."""
    assert main(["frontier", "--data", str(data_file), "--lead-time", "7"]) == 0
    out = capsys.readouterr().out
    assert "7d lead time" in out
    assert "8d protection interval" in out
    assert "mean_on_order" in out


def test_frontier_rejects_a_negative_lead_time_before_it_trains_anything(
    data_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "lightgbm", None)
    assert main(["frontier", "--data", str(data_file), "--lead-time", "-1"]) == 1
    assert "error: --lead-time cannot be negative" in capsys.readouterr().err


def test_frontier_can_price_the_calibrated_model(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of calibrating is that the decision layer gets to use it."""
    argv = ["frontier", "--data", str(data_file), "--model", "gbm_conformal"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "gbm_conformal" in out
    assert "Cheapest at quantile" in out


def test_calibration_scores_the_raw_and_the_calibrated_model_side_by_side(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Printing only the calibrated table would be an advertisement, not a measurement."""
    assert main(["calibration", "--data", str(data_file)]) == 0
    out = capsys.readouterr().out

    assert "trading rows held out" in out
    assert "`gbm_quantile` — coverage against the level it claims" in out
    assert "`gbm_conformal` — coverage against the level it claims" in out
    assert out.count("Worst miss at quantile") == 2


def test_calibration_says_when_a_level_rests_on_a_single_observation(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two stores hold about seventy calibration rows, and 0.99 needs at least ninety-nine.

    Below that the offset for the top level is the worst day that happened rather than an
    estimate of anything, and the command has to say so. This fixture is deliberately the
    small one: the warning is worth nothing if it only appears when it is not needed.
    """
    assert main(["calibration", "--data", str(data_file)]) == 0
    out = capsys.readouterr().out

    assert "rows in the calibration window" in out
    assert "Quantiles 0.99 saturate" in out
    assert "not an estimate of a quantile" in out


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


def test_calibration_breaks_coverage_down_by_store_when_asked(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR 0012. A pooled gap of zero can sit on top of two badly covered stores."""
    assert main(["calibration", "--data", str(data_file), "--by", "store"]) == 0
    out = capsys.readouterr().out

    assert out.count("coverage gap by segment") == 2
    assert out.count("Worst group") == 2
    assert "A marginal table cannot see this" in out


def test_calibration_can_break_coverage_down_by_month(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the ADR 0009 cost: a single December, not just a single store."""
    assert main(["calibration", "--data", str(data_file), "--by", "month"]) == 0
    out = capsys.readouterr().out

    assert "coverage gap by segment" in out
    assert "2014-" in out


def test_calibration_stays_marginal_unless_a_grouping_is_asked_for(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calibration", "--data", str(data_file)]) == 0
    assert "coverage gap by segment" not in capsys.readouterr().out


def test_calibration_can_learn_an_offset_per_store_and_says_how_many_qualified(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR 0012's honest outcome on this data: the floor is derived, and nothing clears it.

    A 0.99 in the grid needs 199 trading rows per store and two synthetic stores hold
    about 35 each, so every group falls back to the pooled offset. That is the correct
    answer rather than a disappointing one, and the command has to report it instead of
    appearing to have calibrated per store.
    """
    argv = ["calibration", "--data", str(data_file), "--calibrate-by", "store"]
    assert main(argv) == 0
    out = capsys.readouterr().out

    assert "Per-store calibration needs 199 trading rows per group" in out
    assert "0 group(s) cleared it; 2 fell back to the pooled offset" in out


def test_calibration_says_nothing_about_groups_when_it_did_not_group(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["calibration", "--data", str(data_file)]) == 0
    assert "fell back to the pooled offset" not in capsys.readouterr().out
