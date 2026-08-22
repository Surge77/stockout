"""Paths and the two split constants every command shares."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = _ROOT / "data"
SALARY_PATH = DATA_DIR / "salary.csv"
LOANS_PATH = DATA_DIR / "loans.csv"

#: Trained models land here. `persistence.load` refuses to read from anywhere else,
#: because loading a joblib file executes code from it.
ARTIFACT_DIR = _ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

#: Passed to every estimator that draws random numbers and to `train_test_split`, so
#: two runs of the same command produce the same table.
RANDOM_SEED = 42

#: The held-out fraction. `train_test_split(test_size=0.2)` throughout, as taught.
TEST_SIZE = 0.2


#: The two problems the package solves. Named here because both `datasets` and
#: `registry` key off it and neither should own the vocabulary.
Task = Literal["regression", "classification"]
