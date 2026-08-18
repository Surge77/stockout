# Glossary

Terms that mean something specific here, or that are routinely used loosely elsewhere.

## Forecasting

**Forecast origin** — the moment the forecast is made. Everything before it is known;
everything after is not. Nearly every leakage bug is a feature that quietly reaches past it.

**Horizon** — how far ahead the forecast reaches, in days. Set by the business (supplier
lead time), not chosen by the modeller. The default here is 7.

**Rolling origin** — a backtest that walks the forecast origin forward through history,
refitting at each step. `evaluate/backtest.py` is the only place it happens, and it is the
only path that produces MASE.

**Expanding vs sliding window** — expanding keeps every fold anchored at the first
observation; sliding keeps a fixed-width training window. Expanding is the default; sliding
is better when there is an early regime change worth forgetting.

**Gap** — days deliberately left unused between train and test. Defaulted to the horizon
on every command that holds a window out, because a holdout whose training rows end the day
before its test rows begin is scoring a one-day forecast however long the horizon claims to
be.

**Time holdout** — one cut on the calendar: everything before a date trains, everything
after it tests. What `compare` and `leakage` use, so that the only thing varying between
two rows of a results table is the estimator.

**Seasonal naive** — predict the same weekday last week. No fitting, no parameters, and on
retail data a genuinely strong forecast. The baseline everything here is measured against.

## Metrics

**WMAPE** — weighted mean absolute percentage error, `sum|error| / sum|actual|`. One
division, at the end. Primary metric here.

**MASE** — mean absolute scaled error. In this repository, model MAE over seasonal-naive
MAE **on the same evaluation window**, so below 1 means the baseline was beaten and 1.000
means it *is* the baseline. Classical MASE scales by in-sample naive MAE instead;
[ADR 0005](decisions/0005-wmape-and-mase-not-mape.md) explains the departure.

**RMSPE** — root mean squared percentage error, ignoring zero actuals. Kaggle's official
Rossmann metric, kept for comparability.

**R²** — the share of variance a model explains, and the metric this project reports
because readers expect it rather than because it decides anything. Pooled across stores,
most of the variance in `sales` is variance *between* stores rather than within them, so a
model that learns only "store 262 is busy and store 307 is quiet" scores well on it and
forecasts nothing. `dummy` scores it at approximately zero by construction — R² *is* the
improvement on predicting the training mean — and can score below zero on a later window,
which means the mean itself has moved.

**Macro-F1** — the per-class F1 scores averaged without weighting by class size. The
metric the classification tables lead with, because a model that drops the smallest class
entirely loses a full third of it while accuracy barely notices.

**Adjacent accuracy** — the share of predictions that are right or one class out. Not a
standard metric, and labelled as such wherever it appears. It exists because Low/Medium/High
are *ordered* and no standard classification metric knows that: predicting High when the
truth is Low is a worse mistake than predicting Medium, and F1 charges the same for both.

**Class balance** — the share of labelled rows in each class. Thirds by construction on the
window the cut points were fitted on, and drifted on any later window. The drift is the
finding rather than an inconvenience: it is why macro-F1 leads and why every classifier is
class-weighted.

## Models and the comparison

**Registry** — `models/registry.py`, one list per task. The CLI's `--model`, the notebook
and the comparison table all read from it, so they cannot drift into disagreeing about
which models exist.

**Registry order** — simplest first, and the results table keeps it. A table sorted by
score answers *which won*; this order also answers *did the extra complexity pay*, which is
the more useful question and the harder one to fake.

**Model spec** — a factory plus the facts a reader needs to interpret the score beside it:
whether the model's categoricals were dummy-trapped, whether its training rows were capped,
and why. Deliberately not an estimator, so it can be a module-level constant reused across
folds without carrying state.

**Subsample cap** — a declared limit on training rows, set only where an estimator's cost
makes the full set impossible rather than merely slow. It travels with the model into the
results table as the `train_rows` column, because a model fitted on 5,000 rows and one
fitted on 800,000 are not comparable and the table must not imply they are
([ADR 0015](decisions/0015-a-subsample-is-reported-never-silent.md)).

**Dummy-variable trap** — keeping every level of a one-hot column alongside an intercept,
which makes a linear design matrix singular. `drop_first` avoids it, and is applied per
model: a tree loses a usable split by dropping a level, and a penalised linear model
resolves the collinearity through its penalty anyway.

**Demand class** — Low, Medium or High, defined as the terciles of **that store's own**
trading-day sales, fitted on the **training window only**. Global cut points would let a
classifier score 0.81 macro-F1 by learning which store it is looking at
([ADR 0016](decisions/0016-demand-classes-are-per-store-terciles-fitted-on-the-training-window.md)).
A closed day gets no class at all — a shut shop has no demand, not low demand.

**Grid** — a search space, held in `models/grids.py` as literals and nothing else. Every
key is a path through the assembled pipeline, so `estimate__alpha` is the estimator's own
`alpha` and `estimate__regressor__C` reaches through a `TransformedTargetRegressor` to the
model inside it.

**Optimism** — `internal − future`: how much a validation protocol over-reported its own
error. Not a standard term; it is that subtraction and nothing more. A protocol with an
optimism near zero is honest about itself even if its absolute score is mediocre
([ADR 0020](decisions/0020-the-leakage-experiment-holds-the-future-fixed.md)).

**Artifact** — one file holding everything needed to answer a prediction request: both
fitted pipelines, the demand thresholds, the horizon, the feature columns and the
provenance. A pipeline without its thresholds can produce a number and no label, because
the label is a property of the training window rather than of the model.

## Leakage

**Leakage** — using information at training time that would not exist at prediction time.
Four flavours are guarded against or priced separately:

- **Split leak** — training on data later than the test set. A shuffled split.
- **Feature leak** — a rolling window that includes its own target row, or a lag shorter
  than the horizon.
- **Column leak** — a column present in the training file and absent at forecast time.
  `customers` is the canonical example.
- **Preprocessing leak** — an imputer, scaler or encoder fitted before the split, so the
  model never sees a test *row* but does see the test set's mean, variance and median. The
  reason every transformer here lives inside a `Pipeline`.

**Denylist** — `features.build.FEATURE_DENYLIST`, the columns a model may never see. It
holds `customers`, the date, the raw holiday string, and **all three** target columns —
because `demand_class_code` is a tercile of `sales`, and "not this task's target" is too
weak a rule when two targets encode each other.

**Future-known covariate** — a variable whose future values are already decided and
recorded, such as a promotion calendar or a trading calendar. Using it is *not* leakage,
and confusing it with a lagged observation is the most common error in this problem.

**Date-major** — a frame sorted `(date, store)`. Required by anything that slices on row
*position*, which `TimeSeriesSplit` does. This package's own default order is `(store,
date)` — right for lag construction, and wrong for a positional splitter, which handed a
store-major frame will train on one set of stores and test on another while producing an
entirely plausible score. `split/strategies.py` refuses a frame that is not date-major.
