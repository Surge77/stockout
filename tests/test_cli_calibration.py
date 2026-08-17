"""The `calibration` command: marginal coverage, per-group coverage, and the modes.

Split from `test_cli.py`. The marginal table is always printed; `--by` adds the
conditional grid of ADR 0012, `--calibrate-by` groups the correction itself, and
`--no-refit` swaps in the estimator ADR 0013 describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout.cli import main


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


def test_calibration_can_serve_the_model_its_offsets_describe(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR 0013, and the sentence it insists on: one assumption gone, one still standing."""
    assert main(["calibration", "--data", str(data_file), "--no-refit"]) == 0
    out = capsys.readouterr().out

    assert "Served without a refit" in out
    assert "the split-conformal theorem applies to it" in out
    assert "Coverage is still not proven" in out
    assert "exchangeable" in out


def test_calibration_claims_no_theorem_when_it_refitted(
    data_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default must not inherit a guarantee it does not have."""
    assert main(["calibration", "--data", str(data_file)]) == 0
    assert "split-conformal theorem applies" not in capsys.readouterr().out
