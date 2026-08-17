"""The registry contract, and one fit-and-predict pass over every model in it.

`spec.py`, `linear.py`, `trees.py`, `kernels.py` and `classifiers.py` are declarations
rather than behaviour — each is a tuple of `ModelSpec` and a handful of factory functions.
Testing them one file at a time would assert that a literal is the literal it is. What is
worth asserting is the contract they all share, so it is asserted here once, over every
entry, and a new model added to any of those files is covered the moment it is registered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockout.config import RANDOM_SEED
from stockout.data import schemas as s
from stockout.data.loaders import merge_store
from stockout.data.synth import make_sales
from stockout.data.synth_stores import make_stores
from stockout.errors import BacktestError
from stockout.features.build import build_features
from stockout.features.store_features import add_store_features
from stockout.models.adapter import SklearnClassifier
from stockout.models.registry import CLASSIFIERS, REGRESSORS, build, model_names, spec_for
from stockout.models.spec import ModelSpec
from stockout.split.strategies import date_major, time_holdout
from stockout.targets import DEMAND_CLASS_CODE, add_demand_class, fit_thresholds

HORIZON = 7


@pytest.fixture(scope="module")
def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small featurised, labelled train/test pair — built once for the whole module.

    Module-scoped because building it is the expensive part and every test below reads
    it without writing to it. The models are still constructed fresh per test.
    """
    frame = merge_store(make_sales(n_stores=4, days=420), make_stores(n_stores=4))
    frame = build_features(add_store_features(frame), horizon=HORIZON)
    frame = date_major(frame).dropna(subset=["sales_lag_28"])

    train, test = time_holdout(frame, test_days=28, gap_days=HORIZON)
    thresholds = fit_thresholds(train)
    return add_demand_class(train, thresholds), add_demand_class(test, thresholds)


ALL_SPECS = [*REGRESSORS, *CLASSIFIERS]


def _classifier(name: str) -> SklearnClassifier:
    """`build` returns either wrapper; only one of them has `predict_proba`."""
    model = build(name, task="classification")
    assert isinstance(model, SklearnClassifier)
    return model


# --- the contract every entry shares ------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: f"{spec.task[:5]}-{spec.name}")
def test_every_spec_builds_an_estimator_from_a_seed(spec: ModelSpec) -> None:
    estimator = spec.build(RANDOM_SEED)
    assert hasattr(estimator, "fit")
    assert hasattr(estimator, "predict")


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: f"{spec.task[:5]}-{spec.name}")
def test_every_spec_explains_why_it_is_in_the_comparison(spec: ModelSpec) -> None:
    """The note is printed beside the score. A model with no reason to be there is noise."""
    assert spec.note


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_names_are_unique_within_a_task(task: str) -> None:
    names = model_names(task)  # type: ignore[arg-type]
    assert len(names) == len(set(names))


def test_the_two_tasks_may_reuse_a_name() -> None:
    """`random_forest` is both. Putting the task in the name would repeat it everywhere."""
    assert set(model_names("regression")) & set(model_names("classification"))


def test_both_tasks_start_from_a_baseline() -> None:
    """Without a floor, neither an R2 of 0.88 nor an accuracy of 71% can be read."""
    assert model_names("regression")[0] == "dummy"
    assert model_names("classification")[0] == "dummy"


def test_an_unknown_model_names_the_alternatives() -> None:
    with pytest.raises(BacktestError, match="choose from"):
        spec_for("gradient_bosting", task="regression")


def test_an_unknown_task_is_rejected() -> None:
    with pytest.raises(BacktestError, match="unknown task"):
        model_names("clustering")  # type: ignore[arg-type]


def test_building_twice_gives_two_objects() -> None:
    """`backtest` builds one per fold; a shared fitted object leaks the previous fold."""
    assert build("ridge", task="regression") is not build("ridge", task="regression")


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: f"{spec.task[:5]}-{spec.name}")
def test_a_capped_model_declares_a_positive_cap(spec: ModelSpec) -> None:
    assert spec.sample_rows is None or spec.sample_rows > 0


def test_the_kernel_models_are_capped_and_the_linear_ones_are_not() -> None:
    """SVR is between quadratic and cubic in the rows; Ridge is not. ADR 0015."""
    assert spec_for("svr", task="regression").sample_rows is not None
    assert spec_for("ridge", task="regression").sample_rows is None


def test_a_spec_needs_a_name() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        ModelSpec(name="", task="regression", build=lambda _seed: None)


def test_a_cap_of_zero_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_rows"):
        ModelSpec(name="x", task="regression", build=lambda _seed: None, sample_rows=0)


# --- every model actually fits and predicts -------------------------------------------------


@pytest.mark.parametrize("name", model_names("regression"))
def test_every_regressor_fits_and_predicts_one_value_per_row(
    name: str, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = split
    predicted = build(name, task="regression", horizon=HORIZON).fit(train).predict(test)
    assert len(predicted) == len(test)
    assert predicted.index.equals(test.index)
    assert predicted.notna().all()


@pytest.mark.parametrize("name", model_names("regression"))
def test_no_regressor_predicts_negative_demand(
    name: str, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """A linear model extrapolates below zero, and a negative sale is not a thing."""
    train, test = split
    assert (build(name, task="regression").fit(train).predict(test) >= 0).all()


@pytest.mark.parametrize("name", model_names("regression"))
def test_every_regressor_predicts_zero_for_a_closed_store(
    name: str, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = split
    predicted = build(name, task="regression").fit(train).predict(test)
    assert (predicted[test[s.OPEN] == 0] == 0.0).all()


@pytest.mark.parametrize("name", model_names("classification"))
def test_every_classifier_predicts_a_valid_class_for_every_trading_row(
    name: str, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, test = split
    predicted = build(name, task="classification").fit(train).predict(test)
    trading = predicted[test[s.OPEN] == 1]
    assert trading.notna().all()
    assert set(trading.unique()) <= {0, 1, 2}


@pytest.mark.parametrize("name", model_names("classification"))
def test_no_classifier_labels_a_closed_day(
    name: str, split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """A shut store has no demand class, so inventing one answers an ungradeable question."""
    train, test = split
    predicted = build(name, task="classification").fit(train).predict(test)
    assert predicted[test[s.OPEN] == 0].isna().all()


# --- the reproducibility and subsampling promises ----------------------------------------------


def test_the_same_seed_produces_the_same_predictions(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """To floating-point tolerance, not bit for bit — and the difference is not the seed.

    A forest with `n_jobs=-1` averages its trees across threads, and floating-point
    addition is not associative, so the order the partial sums happen to complete in
    changes the last bits of the mean. Measured here at ~4e-16 relative, which is machine
    epsilon: the same answer, summed in a different order.

    Asserting `.equals` would make this test fail intermittently and send somebody
    hunting for an unseeded estimator that does not exist. Asserting `allclose` with a
    tight tolerance still catches a genuinely unseeded model, which would differ in the
    first significant figure rather than the sixteenth.
    """
    train, test = split
    first = build("random_forest", task="regression", seed=1).fit(train).predict(test)
    second = build("random_forest", task="regression", seed=1).fit(train).predict(test)
    assert np.allclose(first.to_numpy(), second.to_numpy(), rtol=1e-12, atol=0.0)


def test_a_different_seed_produces_visibly_different_predictions(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The partner to the test above: without it, `allclose` could pass on a constant."""
    train, test = split
    first = build("random_forest", task="regression", seed=1).fit(train).predict(test)
    other = build("random_forest", task="regression", seed=99).fit(train).predict(test)
    assert not np.allclose(first.to_numpy(), other.to_numpy(), rtol=1e-12, atol=0.0)


def test_a_capped_model_reports_the_rows_it_actually_saw(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The cap travels into the results table; a silent subsample is a fiddle."""
    train, _ = split
    spec = spec_for("svr", task="regression")
    model = build("svr", task="regression").fit(train)
    assert model.training_rows <= (spec.sample_rows or len(train))


def test_an_uncapped_model_trains_on_every_trading_row(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = split
    model = build("ridge", task="regression").fit(train)
    assert model.training_rows == int((train[s.OPEN] == 1).sum())


def test_a_classifier_trains_only_on_labelled_rows(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = split
    model = build("logistic", task="classification").fit(train)
    assert model.training_rows == int(train[DEMAND_CLASS_CODE].notna().sum())


# --- failure modes -------------------------------------------------------------------------------


def test_predicting_before_fitting_says_so() -> None:
    with pytest.raises(BacktestError, match="not been fitted"):
        build("ridge", task="regression").predict(pd.DataFrame({s.OPEN: [1]}))


def test_asking_for_features_before_fitting_says_so() -> None:
    with pytest.raises(BacktestError, match="not been fitted"):
        _ = build("ridge", task="regression").features


def test_fitting_on_a_frame_with_no_trading_rows_says_so(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = split
    shut = train.assign(**{s.OPEN: 0})
    with pytest.raises(BacktestError, match="no trading rows"):
        build("ridge", task="regression").fit(shut)


def test_a_model_without_probabilities_says_so_rather_than_faking_them(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """`SVC` is fitted without `probability=True`, which would cost a second fit."""
    train, test = split
    model = _classifier("svc").fit(train)
    with pytest.raises(BacktestError, match="does not produce probabilities"):
        model.predict_proba(test)


def test_a_model_with_probabilities_returns_one_row_per_trading_day(
    split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = split
    model = _classifier("logistic").fit(train)
    probabilities = model.predict_proba(test)
    assert len(probabilities) == int((test[s.OPEN] == 1).sum())
    assert probabilities.shape[1] == 3
