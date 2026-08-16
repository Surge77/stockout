# Architecture

One direction of flow, and three points where it refuses to continue.

```
data/raw/train.csv          data.synth.make_sales
(kaggle, gitignored)        (deterministic, offline)
        |                            |
        +-------------+--------------+
                      v
              data.loaders.read_sales      rename, coerce dtypes, sort
                      v
              data.validate.validate_sales  << GUARD 1: schema + invariants
                      v
              features.build.build_features << GUARD 2: horizon-aware lags
                      |                                  + FEATURE_DENYLIST
                      v
              split.rolling.rolling_origin  << GUARD 3: no fold sees its future
                      v
              evaluate.backtest.backtest    fit per fold, score open days only
                      v
              evaluate.report.to_markdown   a table, and a verdict line
                      v
              models.conformal              offsets so a stated level is the achieved one
                      v
              evaluate.metrics.coverage_table   nominal against empirical, per level
                      v
              inventory.policy              critical ratio -> the quantile to stock to
                      v
              inventory.simulate.frontier   cost and fill rate per service level
                      v
              evaluate.report.frontier_to_markdown   names the cheapest level
```

The second half of that chain is what makes the project an argument rather than a score.
`backtest` answers "how wrong is the forecast"; `frontier` answers "what does being that
wrong cost", and those two questions order the models differently.

The calibration step sits between them because it is the join. `frontier` prices a
*service level*, and a service level the model does not actually deliver prices a policy
nobody chose — so coverage has to be measured before any cost table is believed. On the
committed sample, correcting it moves the cheapest level by a whole grid step.

## The three guards

Each catches a different class of mistake, and each fails differently on purpose.

**Guard 1 — `data/validate.py`.** Structural assertions: required columns, datetime dtype,
unique `(store, date)`, `open` binary, `open == 0 ⟹ sales == 0`, no negative sales. Raises
`SchemaError`. Cheap, and run before anything is modelled.

**Guard 2 — `features/lags.py` and `features/build.py`.** A lag shorter than the horizon
cannot exist at the forecast origin, so `add_lags` raises `LeakageError` at call time.
`add_rolling` shifts by the horizon *before* rolling, so no window contains its own target.
`FEATURE_DENYLIST` keeps out columns that exist in training data and not at forecast time —
`customers` above all, which correlates with `sales` at about 0.9 and is unknowable six
weeks ahead.

**Guard 3 — `split/rolling.py`.** No shuffled splitter exists in the package. `split_frame`
calls `assert_no_leakage` on every fold it produces.

The distinction matters: guards 1 and 2 are *static* — they can be checked by reading a
call. Guard 3 is *relational* between two frames, so it needs data, which is why it is
also a test.

## Layering rule

`data/` and `models/gbm.py` are the only modules that touch the outside world (a
filesystem, Kaggle, LightGBM). Everything between them is pure: same input, same output,
no I/O. That is what lets `features/`, `split/`, `evaluate/` and the baselines be tested
exhaustively with no fixtures beyond a generated frame.

## Module map

```
src/stockout/
├── config.py          paths, horizons, the newsvendor cost pair, env knobs
├── errors.py          StockoutError and its four children
├── cli.py             fetch | synth | describe | backtest | calibration | frontier
├── plots.py           figure styling and saving; no chart builders
├── data/
│   ├── download.py    Kaggle fetch; the only network call in the package
│   ├── synth.py       deterministic generator — every test runs on this
│   ├── schemas.py     every assumption about the file's shape, in one place
│   ├── loaders.py     read, rename, coerce, sort
│   └── validate.py    guard 1
├── features/
│   ├── calendar.py    future-known covariates; takes no horizon, and says why
│   ├── lags.py        guard 2 — shift before you roll, lag >= horizon
│   └── build.py       the matrix, and FEATURE_DENYLIST
├── split/rolling.py   guard 3 — rolling origin, and nothing else
├── models/
│   ├── base.py        Forecaster protocol; zero_when_closed; open_rows
│   ├── baselines.py   naive_last, seasonal_naive, moving_average
│   ├── gbm.py         point (tweedie) and quantile; LightGBM imported inside fit()
│   ├── conformal.py   split-conformal offsets; a probe fit and a deployed fit
│   └── __init__.py    the name -> factory registry the CLI's --model reads
├── evaluate/
│   ├── metrics.py     WMAPE, MASE, RMSPE, pinball, coverage, coverage_table. No MAPE
│   ├── backtest.py    the rolling-origin loop; fresh model per fold
│   └── report.py      markdown tables + verdicts that name the loser and the worst miss
└── inventory/
    ├── policy.py      critical_ratio, the base-stock level, the order it implies
    └── simulate.py    the day-by-day walk, the delivery pipeline, the efficient frontier
```

Every file is under the 300-line limit. `evaluate/metrics.py` and `features/lags.py` are
the two closest to it; split them by responsibility before adding to either.
