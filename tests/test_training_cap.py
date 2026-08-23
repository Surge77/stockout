"""The one model that cannot be fitted on a real-sized split, and what the cap costs.

Kernel `SVC` trains at quadratic cost. Fifty seconds on the synthetic split's 37,000 rows
becomes hours on Rossmann's ~800,000, which is not a table anybody waits for. Dropping it
would have been simpler and worse: `svc` is a *taught* model, and a comparison that omits
the slow taught entry while keeping the fast untaught ones is not the comparison it claims
to be.

So it is capped and the table says so. These tests pin both halves of that bargain — that
the cap bites where it must, and that it is invisible everywhere else, including on the
synthetic figures already published.

`ModelSpec` is frozen and the registry is shared, so the tests that need a smaller cap
patch `evaluate.spec_for` rather than writing through to the real entry. Mutating it would
leak into every test that ran afterwards.
"""

from __future__ import annotations

import dataclasses

import pytest

from stockout import evaluate as evaluate_module
from stockout.datasets import prepare
from stockout.evaluate import evaluate, split, training_rows
from stockout.models import SVC_MAX_ROWS
from stockout.registry import spec_for, specs_for

SMALL_CAP = 500


@pytest.fixture(scope="module")
def data():
    return split(prepare("synthetic"), "classification")


@pytest.fixture
def capped_at_small(monkeypatch: pytest.MonkeyPatch):
    """`svc` capped at `SMALL_CAP`, without touching the shared registry entry."""
    shrunk = dataclasses.replace(
        spec_for("svc", task="classification"), max_training_rows=SMALL_CAP
    )

    def fake(name: str, *, task: str):
        if name == "svc":
            return shrunk
        return spec_for(name, task=task)  # type: ignore[arg-type]

    monkeypatch.setattr(evaluate_module, "spec_for", fake)


def test_only_svc_caps() -> None:
    """A cap on anything else would silently change a published number."""
    capped = [
        spec.name
        for task in ("regression", "classification")
        for spec in specs_for(task)  # type: ignore[arg-type]
        if spec.max_training_rows is not None
    ]
    assert capped == ["svc"]


def test_the_cap_is_inert_on_the_synthetic_split(data) -> None:
    """The synthetic split is smaller than the cap, so every published synthetic figure
    is unchanged by the cap existing. If this fails, the README needs rerunning."""
    assert len(data.X_train) < SVC_MAX_ROWS

    X_fit, y_fit = training_rows("svc", "classification", data)
    assert len(X_fit) == len(data.X_train)
    assert len(y_fit) == len(data.Y_train)


def test_an_uncapped_model_always_gets_every_row(data) -> None:
    X_fit, _ = training_rows("random_forest", "classification", data)
    assert len(X_fit) == len(data.X_train)


@pytest.mark.usefixtures("capped_at_small")
def test_the_cap_bites_once_the_split_is_larger(data) -> None:
    """Rossmann's split is twenty times the real cap. Simulated by lowering the bar
    rather than by downloading a gigabyte; the branch under test is the same one."""
    assert len(data.X_train) > SMALL_CAP

    X_fit, y_fit = training_rows("svc", "classification", data)
    assert len(X_fit) == SMALL_CAP
    assert len(y_fit) == SMALL_CAP


@pytest.mark.usefixtures("capped_at_small")
def test_the_sample_keeps_features_and_labels_aligned(data) -> None:
    """The bug this guards: sampling `X` and `y` independently trains every capped model
    on scrambled labels, and would still produce a plausible-looking accuracy."""
    X_fit, y_fit = training_rows("svc", "classification", data)

    assert X_fit.index.equals(y_fit.index)
    assert y_fit.tolist() == data.Y_train.loc[X_fit.index].tolist()


@pytest.mark.usefixtures("capped_at_small")
def test_the_sample_is_reproducible(data) -> None:
    first, _ = training_rows("svc", "classification", data, seed=42)
    again, _ = training_rows("svc", "classification", data, seed=42)
    other, _ = training_rows("svc", "classification", data, seed=7)

    assert first.index.equals(again.index)
    assert not first.index.equals(other.index)


@pytest.mark.usefixtures("capped_at_small")
def test_a_capped_model_is_still_scored_on_the_whole_test_half(data) -> None:
    """The cap changes how much a model learnt from, never what it is judged against."""
    row = evaluate("svc", "classification", data)

    assert row["n_fit"] == SMALL_CAP
    assert len(data.X_test) > SMALL_CAP
    assert 0.0 <= float(row["accuracy"]) <= 1.0  # type: ignore[arg-type]


def test_every_comparison_row_reports_what_it_was_fitted_on(data) -> None:
    """`n_fit` is on every row, not only the capped one. A lone asterisk is missable."""
    row = evaluate("logistic", "classification", data)
    assert row["n_fit"] == len(data.X_train)
