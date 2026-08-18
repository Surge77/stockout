# 0017 — Every hyperparameter is searched, not defaulted

## Situation

`alpha=1.0` in `linear.py`, `n_neighbors=15` in `kernels.py`, `max_depth=12` in
`trees.py`. Written down as literals, each of them is a magic number, and the honest
answer to "why that value" is "it is what the library ships" or "it looked about right".

That is a bad answer twice over. It is bad in a viva, where the question is asked exactly
because it separates someone who ran a tutorial from someone who chose. And it is bad in
practice, because a default is tuned for the library's median user and this data is not
median: fifty correlated calendar columns, a target in the thousands, and a strong weekly
cycle.

The obvious fix — `GridSearchCV` with `cv=5` — introduces a worse problem than it solves.
sklearn's default `cv` is `KFold`, which shuffles. A search that selects hyperparameters
against shuffled folds picks the model that best predicts its own past.

## Decision

`models/grids.py` holds the search spaces as **literals and nothing else**: no imports
beyond typing, no logic, nothing that can fail at runtime. A reviewer asking what `ridge`
was allowed to consider reads eight lines rather than tracing a builder.

`models/tuning.py` searches them, under three constraints:

1. **`cv=TimeSeriesSplit`, never `KFold`**, with `gap` passed through for the same reason
   `features/lags.py` refuses a lag shorter than the horizon.
2. **The search runs on the rows the model would actually fit on**, via the adapter's
   `training_frame` and `unfitted_pipeline` — so a capped model is tuned on its cap
   (ADR 0015).
3. **`error_score="raise"`.** sklearn's default scores a crashed candidate as NaN and
   carries on, so a pipeline broken for half its grid still reports a cheerful
   `best_score_` from the half that worked.

`n_jobs=1` for the search, with parallelism inside the estimators instead. On Windows,
joblib's loky backend pickles the whole design matrix into every worker process; four
workers on a 160 MB matrix is 640 MB of copies plus a re-import of sklearn per task, and
it is routinely slower than one process.

The grids are small on purpose — three to six values spanning orders of magnitude, not
filling a range finely. The difference between `alpha=0.1` and `alpha=1` is worth knowing;
the difference between `0.1` and `0.15` is noise on this data, and a search that cannot be
rerun is not reproducible evidence.

`grid_for` returns an **empty** grid for a model with nothing to tune rather than raising,
so `tune --models linear` reports zero candidates instead of behaving as though the model
were unknown.

## The result on the committed sample, which does not flatter the decision

Held-out 28 days, 7-day gap, 2,395 training rows:

| model | alpha | held-out R² | held-out WMAPE |
|---|---|---|---|
| ridge | 1.0 *(library default)* | **0.8417** | 0.0899 |
| ridge | 10.0 *(what the search chose)* | 0.8398 | **0.0897** |
| ridge | 100.0 | 0.8296 | 0.0915 |
| lasso | 1.0 | **0.8395** | 0.0902 |
| lasso | 100.0 | 0.8270 | **0.0875** |
| lasso | 1000.0 | 0.6385 | 0.1289 |

The search maximises cross-validated R² and picked `alpha=10.0` for ridge. On the held-out
window that is *worse* on R² than the default it replaced, by 0.002. On 2,395 rows the
regularisation path is nearly flat and the search is choosing between values that do not
differ, so what it buys here is provenance rather than accuracy.

Two things in that table are load-bearing anyway. `lasso` at `alpha=1000` collapses to
R² 0.64, which is the grid's upper end doing its job as a bracket — a chosen value with no
worse value on either side of it is a corner solution and cannot be defended. And the two
metrics disagree about ridge, which is the reminder that `best_score` answers the question
the `scoring` argument asked and no other.

## Cost

**A search costs candidates x folds fits.** `elastic_net` is 4 alphas by 3 L1 ratios by 3
folds — 36 fits of a coordinate-descent model on the full matrix. On Rossmann the full
registry is not a coffee break, which is why `--models` takes a subset and why the grids
stay small.

**The searched values are not fed back into the registry.** `linear.py` still says
`alpha=1.0`, and `tune` prints what it found rather than rewriting the source. Closing that
loop automatically would make the registry a build artifact; leaving it open means a reader
must run `tune` to learn that the literal is not the searched answer.

**Three folds, not five.** Fewer than the backtest uses. The point of cross-validating
during a search is to avoid tuning against one accident, not to produce a publishable
score — the honest number comes from the held-out window afterwards.

**`dummy`, `linear` and `polynomial`'s degree are not searched at all.** The first two have
nothing worth searching; the third is fixed at 2 because `PolynomialFeatures(3)` on a
50-column matrix is a memory bomb, which is a constraint rather than a choice and is
recorded as one in `linear.py`.
