"""The bridge between this package's `Forecaster` protocol and scikit-learn's `fit(X, y)`.

`evaluate/backtest.py` hands a model a *frame* and asks for a *series* — `fit(train)` and
`predict(future)` — because a fold is a slice of a calendar, not a pair of arrays. Every
scikit-learn estimator wants `fit(X, y)` with the columns already chosen and encoded.

Roughly forty lines of adapter is what keeps those two facts from becoming an argument.
Without it the rolling-origin loop, the three baselines, the MASE-against-seasonal-naive
comparison and `evaluate/report.py` would all have needed rewriting to speak sklearn, and
the baselines are the only reason any score here means anything.

Two wrappers rather than one, because the two tasks differ in more than a method name:
they read different targets, drop different rows, and return different dtypes. A single
class with a `task` flag would be four `if` statements wearing a trenchcoat.

**Closed days are not trained on.** A shut store sells zero, every model predicts it
perfectly from `open`, and including those rows spends model capacity learning something
the data already states. `predict` puts the zeros back with `zero_when_closed`, so the
caller sees a prediction for every row it asked about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ..config import RANDOM_SEED
from ..data import schemas as s
from ..errors import BacktestError
from ..features.build import feature_columns
from ..features.preprocess import ColumnRoles, column_roles, make_preprocessor
from ..targets import DEMAND_CLASS_CODE, labelled_rows
from .base import open_rows, zero_when_closed

if TYPE_CHECKING:  # pragma: no cover
    from .spec import ModelSpec


def to_float32(matrix: np.ndarray) -> np.ndarray:
    """Halve the design matrix's memory before it reaches an estimator.

    800k rows by fifty columns is 320 MB in float64 and 160 in float32, and `GridSearchCV`
    pickles the whole thing into every worker process. No estimator here is sensitive to
    the seventh decimal place of a standardised feature.

    A module-level function rather than a lambda, because a lambda cannot be pickled and
    the fitted pipeline has to survive `joblib.dump` for the web app to serve it.
    """
    return np.asarray(matrix, dtype="float32")


class _PipelineModel:
    """Shared assembly: choose the columns, build the preprocessor, fit the pipeline."""

    task = "base"
    target = s.SALES

    def __init__(self, spec: ModelSpec, *, horizon: int = 0, seed: int = RANDOM_SEED) -> None:
        self.spec = spec
        self.name = spec.name
        self.horizon = horizon
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.roles: ColumnRoles | None = None
        self.training_rows = 0
        self.fit_seconds = 0.0

    @property
    def features(self) -> list[str]:
        if self.roles is None:
            raise BacktestError(f"{self.name} has not been fitted")
        return self.roles.all_columns

    def _assemble(self, train: pd.DataFrame) -> Pipeline:
        """Column roles come from the *training* frame, so the encoders never see test rows."""
        candidates = [c for c in feature_columns(train) if c != self.target]
        self.roles = column_roles(train, feature_columns=candidates)
        return Pipeline(
            [
                ("preprocess", make_preprocessor(self.roles, drop_first=self.spec.drop_first)),
                ("compact", FunctionTransformer(to_float32, feature_names_out="one-to-one")),
                ("estimate", self.spec.build(self.seed)),
            ]
        )

    def _fitted(self) -> Pipeline:
        if self.pipeline is None:
            raise BacktestError(f"{self.name} has not been fitted")
        return self.pipeline

    def training_frame(self, train: pd.DataFrame) -> pd.DataFrame:
        """The rows this model would actually fit on — subsampling and all.

        Public because `models/tuning.py` has to search over exactly the same rows the
        model would train on. A grid search run on a different row set than the final fit
        chooses hyperparameters for a model nobody ends up building.
        """
        raise NotImplementedError

    def unfitted_pipeline(self, train: pd.DataFrame) -> Pipeline:
        """The pipeline as it would be, before fitting. What `GridSearchCV` clones."""
        return self._assemble(self.training_frame(train))

    def target_values(self, frame: pd.DataFrame) -> np.ndarray:
        """The `y` for this task, as an array."""
        raise NotImplementedError

    def _subsample(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Cap the training rows for estimators that cannot survive the full set.

        SVR and SVC are between quadratic and cubic in the training rows; KNN pays at
        predict time instead. The cap is a declared property of the model rather than
        something the caller remembers to apply, and `spec.sample_rows` is reported next
        to the score so the reader knows what was actually fitted. ADR 0015.
        """
        cap = self.spec.sample_rows
        if cap is None or len(frame) <= cap:
            return frame
        return frame.sample(n=cap, random_state=self.seed).sort_index()


class SklearnForecaster(_PipelineModel):
    """Predicts `sales`. Satisfies the `Forecaster` protocol `backtest` already speaks."""

    task = "regression"
    target = s.SALES

    def training_frame(self, train: pd.DataFrame) -> pd.DataFrame:
        trading = self._subsample(open_rows(train))
        if trading.empty:
            raise BacktestError(f"{self.name}: no trading rows to fit on")
        return trading

    def target_values(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.target].to_numpy(dtype="float64")

    def fit(self, train: pd.DataFrame) -> SklearnForecaster:
        trading = self.training_frame(train)
        self.pipeline = self._assemble(trading)
        self.training_rows = len(trading)
        self.pipeline.fit(trading, self.target_values(trading))
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        trading = open_rows(future)
        predicted = pd.Series(0.0, index=future.index, dtype="float64")
        if not trading.empty:
            # Demand is not negative, and a linear model extrapolating below zero would
            # put a negative row into a table further downstream.
            fitted = np.asarray(self._fitted().predict(trading), dtype="float64")
            predicted.loc[trading.index] = np.clip(fitted, 0.0, None)
        return zero_when_closed(predicted, future)


class SklearnClassifier(_PipelineModel):
    """Predicts the Low/Medium/High class code. Trained only on rows that carry one."""

    task = "classification"
    target = DEMAND_CLASS_CODE

    def training_frame(self, train: pd.DataFrame) -> pd.DataFrame:
        labelled = self._subsample(labelled_rows(train))
        if labelled.empty:
            raise BacktestError(f"{self.name}: no labelled rows to fit on")
        return labelled

    def target_values(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.target].to_numpy(dtype="int64")

    def fit(self, train: pd.DataFrame) -> SklearnClassifier:
        labelled = self.training_frame(train)
        self.pipeline = self._assemble(labelled)
        self.training_rows = len(labelled)
        self.pipeline.fit(labelled, self.target_values(labelled))
        return self

    def predict(self, future: pd.DataFrame) -> pd.Series:
        """Class codes for trading rows; `pd.NA` for closed ones.

        A closed day has no demand class — `targets.add_demand_class` refuses to give it
        one — so returning a code for it would invent an answer to a question nobody can
        grade. Nullable `Int64` rather than a -1 sentinel, because a sentinel eventually
        gets averaged into something.
        """
        trading = open_rows(future)
        predicted = pd.Series(pd.NA, index=future.index, dtype="Int64")
        if not trading.empty:
            codes = np.asarray(self._fitted().predict(trading), dtype="int64")
            predicted.loc[trading.index] = pd.array(codes, dtype="Int64")
        return predicted

    def predict_proba(self, future: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities, for the models that expose them.

        Not every classifier does — `SVC` needs `probability=True` and pays for it with a
        second, internally cross-validated fit. The registry decides; this raises rather
        than silently returning a hard prediction dressed as a distribution.
        """
        estimator: Any = self._fitted()
        if not hasattr(estimator, "predict_proba"):
            raise BacktestError(f"{self.name} does not produce probabilities")
        trading = open_rows(future)
        return pd.DataFrame(estimator.predict_proba(trading), index=trading.index)
