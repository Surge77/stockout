"""The two course datasets, their columns, and which column each task predicts.

Both files come from `ML/Resource Files/Datasets` and were converted from xlsx to csv
once, so reading them needs pandas and nothing else. Every column in both is numeric:
there is no categorical column to encode and no text column to vectorise, which is why
preprocessing in `models.py` is a scaler and stops there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import LOANS_PATH, SALARY_PATH, Task
from .errors import StockoutError


@dataclass(frozen=True)
class Dataset:
    """One csv, its feature columns, and its target column per task."""

    name: str
    path: Path
    features: tuple[str, ...]
    targets: tuple[tuple[Task, str], ...]
    description: str

    def target_for(self, task: Task) -> str:
        for registered, column in self.targets:
            if registered == task:
                return column
        supported = ", ".join(registered for registered, _ in self.targets)
        raise StockoutError(
            f"dataset {self.name!r} has no {task} target; it supports {supported}"
        )

    def tasks(self) -> tuple[Task, ...]:
        return tuple(task for task, _ in self.targets)


SALARY = Dataset(
    name="salary",
    path=SALARY_PATH,
    features=(
        "age",
        "experience_years",
        "education_level",
        "certifications",
        "projects_completed",
        "interview_score",
        "technical_test_score",
        "communication_score",
        "internships",
        "hours_per_week",
    ),
    targets=(("regression", "salary"),),
    description="Candidate profile to expected salary. The assignment's dataset.",
)

LOANS = Dataset(
    name="loans",
    path=LOANS_PATH,
    features=(
        "Age",
        "Monthly_Income",
        "Credit_Score",
        "Employment_Years",
        "Existing_Debt",
        "Property_Value",
    ),
    targets=(("classification", "Loan_Approved"), ("regression", "Loan_Amount")),
    description="Applicant profile to approval (0/1) and to the amount granted.",
)

DATASETS: tuple[Dataset, ...] = (SALARY, LOANS)
DATASET_NAMES: tuple[str, ...] = tuple(dataset.name for dataset in DATASETS)


def dataset_for(name: str) -> Dataset:
    """Look a dataset up by the name `--dataset` accepts."""
    for dataset in DATASETS:
        if dataset.name == name:
            return dataset
    raise StockoutError(
        f"unknown dataset {name!r}; choose from {', '.join(DATASET_NAMES)}"
    )


def load(dataset: Dataset) -> pd.DataFrame:
    """Read the csv and check the columns this package expects are all present."""
    if not dataset.path.exists():
        raise StockoutError(f"{dataset.path} is missing; the repo ships it in data/")

    frame = pd.read_csv(dataset.path)
    expected = set(dataset.features) | {column for _, column in dataset.targets}
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise StockoutError(f"{dataset.path.name} is missing columns: {', '.join(missing)}")
    return frame


def xy(frame: pd.DataFrame, dataset: Dataset, task: Task) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded frame into the feature matrix and the target for one task."""
    return frame[list(dataset.features)], frame[dataset.target_for(task)]


def describe(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """Per column: dtype, nulls, and the five-number summary. What `head`/`isnull` show."""
    summary = frame.describe().transpose()
    summary.insert(0, "dtype", frame.dtypes.astype(str))
    summary.insert(1, "nulls", frame.isnull().sum())
    summary.insert(2, "role", [_role(column, dataset) for column in frame.columns])
    return summary


def _role(column: str, dataset: Dataset) -> str:
    for task, target in dataset.targets:
        if column == target:
            return f"target ({task})"
    return "feature" if column in dataset.features else "unused"
