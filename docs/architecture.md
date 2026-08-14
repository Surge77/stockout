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
              inventory.simulate            << not built yet (P4)
```

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
├── cli.py             fetch | synth | describe | backtest
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
│   └── gbm.py         STUB — point (tweedie) and quantile
├── evaluate/
│   ├── metrics.py     WMAPE, MASE, RMSPE, pinball, coverage. No MAPE
│   ├── backtest.py    the rolling-origin loop; fresh model per fold
│   └── report.py      markdown table + a verdict that names the loser
└── inventory/
    ├── policy.py      critical_ratio implemented; order-up-to STUB
    └── simulate.py    STUB — fill rate, holding cost, efficient frontier
```

Every file is under the 300-line limit. `evaluate/metrics.py` and `features/lags.py` are
the two closest to it; split them by responsibility before adding to either.
