"""The commands that fit models, exercised end to end against synthetic data.

`test_cli.py` covers the cheap surface — the parser, the two data commands and the
baselines. What is here costs real seconds because every one of these commands fits a
scikit-learn pipeline, so each test names the smallest model that can demonstrate the
behaviour rather than the whole registry.

Every test passes `--models` or a named estimator for that reason. A test that runs the
default full comparison would take ten seconds to assert something a two-model run
asserts just as well, and a suite nobody wants to run is a suite that stops being run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stockout import config
from stockout.cli import main


@pytest.fixture
def frame_args(data_file: Path, store_file: Path) -> list[str]:
    """The `--data`/`--stores` pair every model command needs, as argv fragments."""
    return ["--data", str(data_file), "--stores", str(store_file)]


def test_prepare_reports_the_warmup_drop_and_writes_a_cache(
    frame_args: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache = tmp_path / "prepared.parquet"
    assert main(["prepare", *frame_args, "--out", str(cache)]) == 0

    out = capsys.readouterr().out
    assert "rows ready at horizon" in out
    assert "dropped for lag warm-up" in out
    assert cache.exists()


def test_backtest_accepts_a_registered_sklearn_model(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of the registry reaching the command line."""
    assert main(["backtest", *frame_args, "--model", "ridge", "--folds", "3"]) == 0
    out = capsys.readouterr().out
    assert "`ridge` — 3 rolling-origin folds" in out
    assert "MASE" in out


def test_compare_names_the_winner_on_wmape(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", *frame_args, "--models", "dummy", "ridge"]) == 0
    out = capsys.readouterr().out
    assert "wins on wmape" in out
    # The floor has to be in the table for the winner's number to mean anything.
    assert "| dummy |" in out


def test_compare_classification_reports_per_class_recall(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        ["compare", *frame_args, "--task", "classification", "--models", "dummy", "logistic"]
    ) == 0
    out = capsys.readouterr().out
    assert "wins on macro_f1" in out
    assert "recall_low" in out
    assert "recall_high" in out


def test_compare_rejects_an_unknown_model_as_one_line(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Not an argparse `choices` rejection: the valid names depend on `--task`."""
    assert main(["compare", *frame_args, "--models", "xgboost"]) == 1
    assert "unknown regression model" in capsys.readouterr().err


def test_leakage_prints_four_arms_and_an_optimism_column(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["leakage", *frame_args]) == 0
    out = capsys.readouterr().out
    for arm in ("honest", "shuffled split", "preprocessing leak", "future feature"):
        assert arm in out
    assert "optimism" in out


def test_tune_reports_the_candidates_and_what_it_chose(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["tune", *frame_args, "--models", "ridge", "--folds", "2"]) == 0
    out = capsys.readouterr().out
    assert "candidates" in out
    assert "alpha=" in out


def test_tuning_a_model_with_no_grid_is_zero_candidates_not_an_error(
    frame_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """`linear` has nothing to search, which is a result rather than a failure."""
    assert main(["tune", *frame_args, "--models", "linear", "--folds", "2"]) == 0
    assert "| linear |" in capsys.readouterr().out


def test_train_writes_an_artifact_that_predict_reads_back(
    frame_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The round trip, which is the only thing that proves the artifact is complete.

    `ARTIFACT_DIR` is redirected because `persistence.load` refuses a path outside it —
    loading a joblib file runs code from it, so the directory is a guard rather than a
    convention.
    """
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)

    assert main(["train", *frame_args, "--regressor", "ridge", "--classifier", "logistic"]) == 0
    trained = capsys.readouterr().out
    assert "features" in trained
    assert (tmp_path / "model.joblib").exists()

    assert main(["predict", *frame_args, "--store", "1", "--date", "2015-01-02"]) == 0
    predicted = capsys.readouterr().out
    assert "predicted, class" in predicted
    assert "cut points" in predicted


def test_predicting_beyond_the_horizon_is_a_one_line_error(
    frame_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The limit is a property of the lag the model reads, and it is stated, not guessed."""
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    assert main(["train", *frame_args, "--regressor", "ridge", "--classifier", "logistic"]) == 0
    capsys.readouterr()

    assert main(["predict", *frame_args, "--store", "1", "--date", "2016-06-01"]) == 1
    assert "cannot forecast" in capsys.readouterr().err


def test_predict_refuses_a_model_from_outside_the_artifact_directory(
    frame_args: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`joblib.load` executes what it reads, so the path is not a free parameter."""
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path / "artifacts")
    elsewhere = tmp_path / "smuggled.joblib"
    elsewhere.write_bytes(b"not a model")

    assert main(
        ["predict", *frame_args, "--store", "1", "--date", "2015-01-02",
         "--model-path", str(elsewhere)]
    ) == 1
    assert "refusing to load a model from outside" in capsys.readouterr().err


def test_a_missing_store_file_names_the_file_rather_than_dropping_features(
    data_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Carrying on without the join produces a frame that fits and scores worse in silence."""
    assert main(
        ["compare", "--data", str(data_file), "--stores", str(tmp_path / "absent.csv")]
    ) == 1
    assert "store metadata is missing" in capsys.readouterr().err
