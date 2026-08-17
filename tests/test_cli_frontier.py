"""The `frontier` command: what it prices, and what it refuses before training.

Split from `test_cli.py` because these are the expensive ones — every test here fits
six boosters — and because the checks that must happen *before* a fit are only
meaningful next to the ones that need it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout.cli import main


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
