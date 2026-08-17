"""The two things a model is asked to predict, and where their definitions come from.

**Regression** predicts `sales` directly. Nothing to define.

**Classification** predicts whether a day is Low, Medium or High demand — and that phrase
has no meaning until somebody fixes the cut points. Three decisions make it one:

**Per store, not globally.** Global terciles would put every day of a busy store in
"High" and every day of a quiet one in "Low", so the classifier would learn store
identity and score around 95% while knowing nothing. Per-store terciles subtract the
store's own level and leave the question worth asking: *is today busy for this shop?*
`absolute_thresholds` builds the global version too, precisely so the notebook can show
it scoring better and being worth less.

**Fitted on the training window only.** Cut points are a statistic of the data, so
computing them over all rows lets the test set vote on its own labels. This is the same
leak a scaler fitted outside a `Pipeline` causes, and it is harder to see because nothing
about it looks like a model.

**Trading days only.** A closed store sells zero, and including those rows puts the whole
closure mass in the bottom bin — "Low" would come to mean "shut", which is a question
nobody asked and every model can already answer from `open`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEMAND_CLASS_LABELS
from .data import schemas as s
from .errors import SchemaError

#: Re-exported from the schema, which owns them so that `features/build.py` can deny
#: them as features without importing this module.
DEMAND_CLASS = s.DEMAND_CLASS
DEMAND_CLASS_CODE = s.DEMAND_CLASS_CODE

#: Two cut points make three classes. Terciles, so each class holds a third of the
#: training days for its store — which also means accuracy has a 33% floor rather than
#: the 50% a two-class problem would give it.
_QUANTILES: tuple[float, float] = (1 / 3, 2 / 3)


@dataclass(frozen=True)
class DemandThresholds:
    """Where one store's Low/Medium/High boundaries sit, plus a fallback for the rest.

    Persisted alongside the fitted pipeline. Without it a served prediction can produce
    a number and no label, because the label is not a property of the model — it is a
    property of the training window the model was fitted on.
    """

    per_store: dict[int, tuple[float, float]] = field(default_factory=dict)
    fallback: tuple[float, float] = (0.0, 0.0)

    def for_store(self, store: int) -> tuple[float, float]:
        """This store's cut points, or the pooled ones if it had no training rows.

        The fallback is not defensive padding for a case that cannot happen: roughly a
        sixth of Rossmann's stores vanish from the file for a refurbishment quarter, so
        a training window landing inside one leaves that store with nothing to measure.
        """
        return self.per_store.get(store, self.fallback)

    @property
    def stores(self) -> int:
        return len(self.per_store)


def fit_thresholds(train: pd.DataFrame) -> DemandThresholds:
    """Learn per-store tercile cut points from the training window's trading days."""
    trading = _trading_rows(train)
    if trading.empty:
        raise SchemaError("cannot fit demand thresholds: no trading days in the training window")

    per_store: dict[int, tuple[float, float]] = {}
    for store, values in trading.groupby(s.STORE, observed=True)[s.SALES]:
        if len(values) == 0:
            continue
        low, high = np.quantile(values.to_numpy(dtype="float64"), _QUANTILES)
        per_store[int(str(store))] = (float(low), float(high))

    pooled = np.quantile(trading[s.SALES].to_numpy(dtype="float64"), _QUANTILES)
    return DemandThresholds(per_store=per_store, fallback=(float(pooled[0]), float(pooled[1])))


def absolute_thresholds(train: pd.DataFrame) -> DemandThresholds:
    """One pair of cut points for every store — the version that looks better and is worse.

    Kept because a claim is stronger with its alternative measured beside it. A model
    trained against these labels scores far higher than one trained against per-store
    terciles, and the reason is that it is being graded on recognising which store it is
    looking at. `docs/results.md` reports both.
    """
    fitted = fit_thresholds(train)
    return DemandThresholds(per_store={}, fallback=fitted.fallback)


def add_demand_class(frame: pd.DataFrame, thresholds: DemandThresholds) -> pd.DataFrame:
    """Append `demand_class` (ordered categorical) and `demand_class_code` (0/1/2).

    Both, because two consumers want different things: a confusion matrix wants readable
    labels and an estimator wants integers. Deriving them together is what stops the two
    drifting apart.

    Closed days get no label at all — `pd.NA` — rather than a bottom-bin one. They are
    dropped before training; giving them a class would mean asserting that a shut shop
    has low demand, when what it has is no demand and no observation of it.
    """
    _require_columns(frame)
    out = frame.copy()

    cuts = np.array([thresholds.for_store(int(store)) for store in out[s.STORE]], dtype="float64")
    sales = out[s.SALES].to_numpy(dtype="float64")

    # Comparisons are `>`, so a value sitting exactly on a cut point falls to the lower
    # class. np.quantile puts a cut point on an observed value whenever the count is
    # divisible by three, which for a 365-day window is often.
    ranked = (sales > cuts[:, 0]).astype("int8") + (sales > cuts[:, 1]).astype("int8")
    # -1 is pandas' own "no category" code, so it round-trips into a Categorical as NaN
    # without a second missing-value convention being invented here.
    codes = np.where(_trading_mask(out), ranked, -1).astype("int8")

    out[DEMAND_CLASS_CODE] = pd.Series(codes, index=out.index, dtype="Int64").mask(codes < 0)
    out[DEMAND_CLASS] = pd.Categorical.from_codes(
        codes=codes, dtype=pd.CategoricalDtype(list(DEMAND_CLASS_LABELS), ordered=True)
    )
    return out


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Labels to 0/1/2 in Low, Medium, High order.

    `sklearn.preprocessing.LabelEncoder` is the obvious tool and is a trap here: it sorts
    its classes, so it would map High->0, Low->1, Medium->2. Every confusion-matrix row,
    every per-class recall and every classification report would then be captioned
    wrongly *and look entirely plausible*. The order comes from `config` instead, and
    `tests/test_targets.py` asserts it rather than trusting it.
    """
    ordered = pd.Categorical(labels, categories=list(DEMAND_CLASS_LABELS), ordered=True)
    if bool(pd.isna(ordered.codes).any()) or bool((ordered.codes < 0).any()):
        raise SchemaError("cannot encode demand labels: a value is not one of Low/Medium/High")
    return np.asarray(ordered.codes, dtype="int64")


def class_balance(frame: pd.DataFrame) -> pd.Series:
    """Share of labelled rows in each class, in Low/Medium/High order.

    Worth printing at every stage. On a training window the three are a third each by
    construction, which makes accuracy look flattering; on a later test window they have
    drifted, and the drift is the finding rather than an inconvenience.
    """
    if DEMAND_CLASS not in frame.columns:
        raise SchemaError(f"{DEMAND_CLASS!r} is missing; call add_demand_class first")
    counts = frame[DEMAND_CLASS].value_counts(normalize=True, sort=False)
    return counts.reindex(list(DEMAND_CLASS_LABELS), fill_value=0.0)


def labelled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows a classifier may train on: trading days that carry a class."""
    if DEMAND_CLASS_CODE not in frame.columns:
        raise SchemaError(f"{DEMAND_CLASS_CODE!r} is missing; call add_demand_class first")
    return frame[frame[DEMAND_CLASS_CODE].notna()].copy()


def _trading_rows(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame)
    return frame[_trading_mask(frame)]


def _trading_mask(frame: pd.DataFrame) -> np.ndarray:
    if s.OPEN not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    return (frame[s.OPEN] == 1).to_numpy()


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted({s.STORE, s.SALES} - set(frame.columns))
    if missing:
        raise SchemaError(f"demand classes need {missing}")
