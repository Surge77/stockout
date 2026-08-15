# 0003 — Only rolling-origin splitters exist

## Situation

`sklearn.model_selection.train_test_split` defaults to `shuffle=True`. On a time series
that trains on next month to predict last month, and the resulting score is not optimistic
— it is meaningless. The wrong thing is also the shortest thing to type, and it appears in
a large share of published retail-forecasting notebooks.

Documenting "don't do that" does not work. People read the README once and the code every
day.

## Decision

`stockout.split.rolling` exposes rolling-origin and expanding-window splitters and nothing
else. There is no `shuffle` argument and no `random_state` anywhere in the package.

Three guards back it up, each catching a different class of mistake:

| Guard | Catches | Where |
|---|---|---|
| `assert_no_leakage` | A split whose train overlaps its test | `split/rolling.py`, called by `split_frame` |
| Lag-vs-horizon check | A feature that could not exist at the forecast origin | `features/lags.py`, raises `LeakageError` |
| `FEATURE_DENYLIST` | A column unknown at forecast time (`customers`) | `features/build.py` |

The first is relational and needs data, so it is also a test
(`test_split.py::test_no_fold_trains_on_the_future`). The sentinel test in `test_lags.py`
covers the general case: perturbing the newest target must change no feature anywhere.

## Cost

**Backtests are roughly five times more expensive than a single split**, because five
folds means five fits. On this data that is seconds; on a 42,000-series dataset it is not.

**Short histories get awkward.** Five 42-day folds on top of a 365-day minimum training
window needs 575 days of history. `rolling_origin` raises `BacktestError` rather than
quietly returning three folds when five were requested, which is the correct behaviour and
is still an obstacle when experimenting on a subset.
