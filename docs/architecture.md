# Architecture

One direction of flow, and three points where it refuses to continue.

```
data/raw/train.csv + store.csv        data.synth + data.synth_stores
(kaggle, gitignored)                  (deterministic, offline)
        |                                      |
        +-------------------+------------------+
                            v
                    data.loaders.read_sales      rename, coerce dtypes, sort
                    data.loaders.merge_store     left join, so a missing store shows
                            v
                    data.validate.validate_sales << GUARD 1: schema + invariants
                            v
                    features.store_features      store metadata x the row's calendar
                            v
                    features.build.build_features << GUARD 2: horizon-aware lags
                            |                                   + FEATURE_DENYLIST
                            v
                    split.strategies.date_major  positional splitters need date order
                            v
                    dataset.prepare              drop the lag warm-up, fit the labels
                            v
             +--------------+---------------+------------------+
             v              v               v                  v
      split.rolling   split.strategies  models.tuning     targets.fit_thresholds
      rolling_origin  time_holdout      GridSearchCV      per-store terciles,
      << GUARD 3      + time_series_cv  over time folds   training window only
             v              v               v
      evaluate.backtest  evaluate.comparison   evaluate.leakage
      one model,         every model,          four protocols,
      many folds         one holdout           one fixed future
             v              v                    v
             +--------------+--------------------+
                            v
                    evaluate.report              a table, and a verdict line
                            v
                    train.train -> persistence.save -> predict.forecast
```

The chain forks after `dataset.prepare` because three questions need three shapes of
evidence, and running one of them through the others' machinery is how a comparison stops
being a comparison:

- **`backtest`** takes one model across many folds. It answers *how stable is this*, and it
  is the only path that produces MASE, because MASE needs a baseline refitted on each
  training window.
- **`comparison`** takes every model across one holdout. It answers *which one, and did the
  extra complexity pay*, and it must hold the window fixed so the only thing varying
  between two rows is the estimator.
- **`leakage`** takes one model across four *protocols*. It answers *what would each way of
  validating have told me*, which is a question about method rather than about models — so
  the model is held fixed instead.

## The three guards

Each catches a different class of mistake, and each fails differently on purpose.

**Guard 1 — `data/validate.py`.** Structural assertions: required columns, datetime dtype,
unique `(store, date)`, `open` binary, `open == 0 ⟹ sales == 0`, no negative sales. Raises
`SchemaError`. Cheap, and run before anything is modelled.

**Guard 2 — `features/lags.py` and `features/build.py`.** A lag shorter than the horizon
cannot exist at the forecast origin, so `add_lags` raises `LeakageError` at call time.
`add_rolling` shifts by the horizon *before* rolling, so no window contains its own target.
`FEATURE_DENYLIST` keeps out columns that exist in training data and not at forecast time —
`customers` above all — and also the two label columns, because `demand_class_code` is a
tercile of `sales` and "not this task's target" is too weak a rule when two targets encode
each other.

**Guard 3 — `split/rolling.py` and `split/strategies.py`.** `split_frame` calls
`assert_no_leakage` on every fold it produces. `strategies` adds a second, subtler guard:
`TimeSeriesSplit` slices on **row position**, and every frame this package builds arrives
sorted `(store, date)` — store-major — so handed one of those it would train on stores
1..186 and test on 187..372, producing an entirely plausible R² with nothing raising
anywhere. Every splitter there calls `assert_date_major` first.

The distinction matters: guards 1 and 2 are *static* — they can be checked by reading a
call. Guard 3 is *relational* between two frames, so it needs data, which is why it is
also a test.

`split/strategies.py` is the one module allowed to name a shuffled splitter, and
`tests/test_no_random_splits.py` enforces that by parsing the AST of every other one. A
project cannot price what shuffling costs while refusing to import the thing that shuffles.

## The preprocessor is the reason the comparison is fair

Twenty-one models, one `ColumnTransformer`, fitted **inside** each fold. The imputer's
medians, the scaler's means and the encoder's vocabulary are all learned from training rows
only — which is what a `Pipeline` is for, and what `evaluate/leakage.py` prices the absence
of.

| Branch | Columns | Why that treatment |
|---|---|---|
| numeric | everything else | median-impute (`competition_distance` is heavily right-skewed) then standardise, because Ridge, Lasso, KNN and SVM all need it and trees do not care |
| categorical | `store_type`, `assortment`, `state_holiday_type`, `day_of_week` | four, three, four and seven levels; `handle_unknown="ignore"` so an unseen level at serving time is a row of zeros rather than an exception |
| ordinal | `store` | passed through as an integer — 1115 one-hot columns would be 3.77 GB, would make every KNN distance between two stores the same constant, and would be 1115 coefficients nobody can defend ([ADR 0019](decisions/0019-store-is-an-ordinal-id-not-1115-one-hot-columns.md)) |
| text | `promo_interval` | Tf-idf over a **fixed** month vocabulary, so a fold in which nobody promotes still produces the same columns |

Two decisions in that table are about output *shape* rather than about encoding, and they
are the same decision twice: the fixed vocabulary, and `keep_empty_features=True` on the
imputer. A transformer whose output width depends on which rows landed in the fold works
until the fold is one store, and then it either raises from four frames down or silently
drops a feature.

`drop_first` is per model rather than global. Keeping every level alongside an intercept
makes a linear design matrix singular — the dummy-variable trap — while a tree loses a
usable split by dropping one, and a penalised linear model does not care either way.

## Layering rule

`data/` is the only package that touches the outside world — a filesystem, Kaggle — and
`persistence.py` is the only other module that writes. Everything between them is pure:
same input, same output, no I/O. That is what lets `features/`, `split/`, `evaluate/`,
`targets` and the baselines be tested exhaustively with no fixtures beyond a generated
frame.

`models/adapter.py` is the seam. `evaluate/backtest.py` hands a model a *frame* and asks
for a *series*, because a fold is a slice of a calendar; every scikit-learn estimator wants
`fit(X, y)` with the columns already chosen and encoded. Roughly forty lines of adapter is
what stops those two facts becoming an argument — without it the rolling-origin loop, the
three baselines and the MASE comparison would all have needed rewriting to speak sklearn,
and the baselines are the only reason any score here means anything.

## Module map

```
src/stockout/
├── config.py          paths, horizons, the subsample cap, the class labels, env knobs
├── errors.py          StockoutError and its four children
├── cli.py             the argument surface: what each subcommand accepts, and nothing else
├── commands.py        what each subcommand does once the parser has agreed
├── dataset.py         the six preparation steps, in the one order that is correct
├── targets.py         per-store demand terciles; the global version, kept to be beaten
├── train.py           fit both tasks on everything; the scores come from the holdout
├── persistence.py     the artifact: model, thresholds, horizon, provenance
├── predict.py         rebuild the lags around one future day, read the last row
├── plots.py           figure styling and saving; no chart builders
├── data/
│   ├── download.py    Kaggle fetch; the only network call in the package
│   ├── synth.py       deterministic sales generator — every test runs on this
│   ├── synth_stores.py the metadata half, with the missing-value patterns that matter
│   ├── schemas.py     every assumption about the file's shape, in one place
│   ├── loaders.py     read, rename, coerce, sort, join
│   └── validate.py    guard 1
├── features/
│   ├── calendar.py    future-known covariates; takes no horizon, and says why
│   ├── lags.py        guard 2 — shift before you roll, lag >= horizon
│   ├── store_features.py  the three date-relative features the join makes possible
│   ├── preprocess.py  one ColumnTransformer, four branches, fitted inside the fold
│   └── build.py       the matrix, and FEATURE_DENYLIST
├── split/
│   ├── rolling.py     guard 3 — the rolling-origin fold layout
│   └── strategies.py  random / holdout / TimeSeriesSplit, and the date-major guard
├── models/
│   ├── base.py        Forecaster protocol; zero_when_closed; open_rows
│   ├── spec.py        a factory, plus the facts needed to read the score beside it
│   ├── registry.py    one list per task, so CLI, notebook and table cannot drift
│   ├── adapter.py     frame-in/series-out over sklearn's fit(X, y)
│   ├── baselines.py   naive_last, seasonal_naive, moving_average
│   ├── linear.py      the floor, and the three ways of penalising it
│   ├── trees.py       one tree, bagging, a forest, and boosting — for both tasks
│   ├── kernels.py     KNN and the support-vector models: the ones that are capped
│   ├── classifiers.py logistic regression and the two floors
│   ├── grids.py       the search spaces, as literals and nothing else
│   ├── tuning.py      GridSearchCV over TimeSeriesSplit, never KFold
│   └── __init__.py    the one name -> factory surface the CLI's --model reads
└── evaluate/
    ├── metrics.py     WMAPE, MASE, RMSPE, R². Deliberately no MAPE
    ├── classification.py  macro-F1, per-class recall, adjacent accuracy, confusion
    ├── backtest.py    the rolling-origin loop; fresh model per fold
    ├── comparison.py  the results table, with rows fitted and seconds as columns
    ├── leakage.py     four protocols, one fixed future, an optimism column
    └── report.py      markdown tables + verdicts that name the loser
```

Every file is under the 300-line limit. `models/` is eight files rather than one for that
reason and for a better one: `linear.py`, `trees.py` and `kernels.py` each answer a
different question about *why* a model family exists, and a single `models.py` would have
buried three arguments inside one import block.

`evaluate/leakage.py` (239 lines) and `features/preprocess.py` (224) are the closest to the
limit; split them by responsibility before adding to either.
