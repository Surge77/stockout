# stockout

[![CI](https://github.com/Surge77/stockout/actions/workflows/ci.yml/badge.svg)](https://github.com/Surge77/stockout/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Coverage 98%](https://img.shields.io/badge/coverage-98%25-brightgreen)](#testing)
[![Checked with pyright](https://img.shields.io/badge/types-pyright-blue)](https://github.com/microsoft/pyright)

Twenty-one scikit-learn models on one retail forecasting problem, behind one preprocessor,
compared in one table that reports what each of them cost to fit. Validated by
rolling-origin backtesting, with no random split anywhere in the package except the one
module that exists to price what a random split costs you.

Named after the failure it exists to prevent.

```bash
pip install -e ".[dev]"
python -m stockout describe                        # what's in the sample
python -m stockout compare                         # every regressor, simplest first
python -m stockout compare --task classification   # every classifier, against a 33% floor
python -m stockout leakage                         # what four validation protocols believe
python -m stockout tune --models ridge lasso       # where the hyperparameters came from
python -m stockout train && python -m stockout predict --store 1 --date 2015-01-02
```

## Why this project exists

Most beginner forecasting projects are ordinary tabular regression that happens to have a
date column. Shuffle the rows, fit a model, report R². The score is excellent and it is
not a forecast, because the model was shown next month while predicting last month.

Time series breaks the assumption every other tabular project rests on:
`train_test_split(shuffle=True)` stops being a style preference and becomes a **bug**. That
one difference is why this project is worth building and nine near-identical
price-prediction projects are not.

Three things follow from it, and they are the whole content.

**Leakage is caught by the test suite, not by discipline.** Three separate guards, each for
a different class of mistake. A lag shorter than the horizon raises. A column unknown at
forecast time is on a denylist. And the most useful is a property test: perturb the newest
target value and assert that *no feature anywhere changes* — any leak, however written,
breaks it, and its partner test asserts that perturbing an *old* value **does** change
later features, so the first cannot pass by being vacuous.

**A model comparison is only a comparison if one thing varied.** Every model here sits
behind the same `ColumnTransformer`, fitted inside the fold, on the same rows, scored on the
same window. What differs between two rows of the results table is the estimator and
nothing else — and where that is not true, because three models cannot afford every row,
the row count is a printed column rather than a footnote
([ADR 0015](docs/decisions/0015-a-subsample-is-reported-never-silent.md)).

**Every hyperparameter has an answer to "why that value".** `alpha=1.0` written as a
literal is a magic number.
[`models/grids.py`](src/stockout/models/grids.py) holds the search spaces and
[`models/tuning.py`](src/stockout/models/tuning.py) searches them over `TimeSeriesSplit`,
never `KFold` ([ADR 0017](docs/decisions/0017-every-hyperparameter-is-searched-not-defaulted.md)).
On the committed sample the search picks a value marginally *worse* than the library
default, and that is reported too.

> **This used to be a different project.** Up to `v0.4.0` it forecast quantiles, calibrated
> them conformally and priced the stocking decision with a newsvendor inventory simulator.
> That layer was deleted — one model wearing five names, and a cost pair that was invented.
> [ADR 0014](docs/decisions/0014-a-scikit-learn-comparison-not-an-inventory-system.md)
> records the removal and what it gave up. Seven older ADRs describe code that no longer
> ships and say so at the top.

## What's in here

**Twelve regressors and nine classifiers**, in registry order — simplest first, because a
table sorted by score answers *which won* while this order also answers *did the extra
complexity pay*.

| Family | Regression | Classification |
|---|---|---|
| floor | `dummy` | `dummy` |
| linear | `linear`, `ridge`, `lasso`, `elastic_net`, `polynomial` | `logistic` |
| trees | `decision_tree`, `bagging`, `random_forest`, `hist_gradient_boosting` | the same four |
| kernel & instance | `knn`, `svr` | `knn`, `svc`, `linear_svc` |

`bagging` and `random_forest` are both here on purpose: they differ by exactly one idea — a
random feature subset at each split — so the gap between them isolates what that idea is
worth. `linear_svc` is here so the cost of the kernel trick is visible against a model that
solves the same problem in linear time.

**Two targets.** Regression predicts `sales`. Classification predicts whether a day is
Low, Medium or High demand for *that shop* — per-store terciles, fitted on the training
window only, because global cut points let a classifier score 0.81 by learning which store
it is looking at ([ADR 0016](docs/decisions/0016-demand-classes-are-per-store-terciles-fitted-on-the-training-window.md)).

**One preprocessor, four branches.** Numeric columns are median-imputed and standardised;
four low-cardinality categoricals are one-hot encoded; `promo_interval` goes through a
Tf-idf over a fixed month vocabulary; and `store` is passed through as an integer rather
than becoming 1115 one-hot columns and 3.77 GB
([ADR 0019](docs/decisions/0019-store-is-an-ordinal-id-not-1115-one-hot-columns.md)).

**In the notebook** — the analysis. The five questions are committed in
[docs/questions.md](docs/questions.md) *before* any chart exists, so findings cannot be
retrofitted to whatever turned up ([ADR 0006](docs/decisions/0006-analysis-in-notebooks.md)).
Every cell is run by `tests/test_notebook.py`.

**Not here** — the real data. Every number below was produced by a generator, which is one
`python -m stockout fetch` and two changed lines at the top of the notebook away. See
[Known limits](#known-limits).

## Current state

`python -m stockout compare` on the committed synthetic sample — 4 stores, 730 days, a
28-day held-out window with a 7-day gap, scored on trading rows only. 2,279 training rows,
92 scored.

| Model | R² | WMAPE | MAE | RMSE | seconds |
|---|---|---|---|---|---|
| `dummy` | −0.003 | 0.229 | 1,703 | 2,261 | 0.1 |
| `linear` | 0.835 | 0.090 | 669 | 918 | 0.0 |
| `ridge` | 0.842 | **0.090** | 669 | 898 | 0.0 |
| `lasso` | 0.840 | 0.090 | 672 | 904 | 0.1 |
| `elastic_net` | 0.779 | 0.095 | 708 | 1,060 | 0.0 |
| `polynomial` | **0.845** | 0.093 | 691 | 888 | 0.3 |
| `decision_tree` | 0.780 | 0.100 | 748 | 1,058 | 0.0 |
| `bagging` | 0.823 | 0.095 | 709 | 949 | 7.7 |
| `random_forest` | 0.823 | 0.095 | 709 | 949 | 0.2 |
| `hist_gradient_boosting` | 0.843 | 0.091 | 674 | 893 | 1.9 |
| `knn` | 0.762 | 0.105 | 778 | 1,101 | 0.1 |
| `svr` | 0.833 | 0.090 | 670 | 923 | 0.1 |

**The headline is that the ladder is flat, and that is the finding.** A straight line
scores WMAPE 0.0899. The best model in the table scores 0.0899. `bagging` spends 7.7
seconds to land below both. On a generator whose promotion calendar and weekday pattern are
deterministic, a linear model on the right features is the correct answer, and reporting
otherwise would be inventing a result. `dummy` at R² −0.003 is the definition of R² working:
it predicts the training mean, and R² *is* the improvement on that.

The classification half separates more, because its floor is higher and its target is
harder:

| Model | accuracy | macro-F1 | adjacent | recall Low | recall Med | recall High | seconds |
|---|---|---|---|---|---|---|---|
| `dummy` | 0.304 | 0.156 | 0.641 | 1.000 | 0.000 | 0.000 | 0.1 |
| `logistic` | 0.685 | 0.691 | 1.000 | 0.786 | 0.613 | 0.667 | 0.2 |
| `decision_tree` | 0.576 | 0.579 | 0.989 | 0.821 | 0.516 | 0.424 | 0.1 |
| `bagging` | 0.565 | 0.567 | 1.000 | 0.821 | 0.548 | 0.364 | 0.4 |
| `random_forest` | 0.630 | 0.638 | 1.000 | 0.786 | 0.581 | 0.545 | 0.2 |
| `hist_gradient_boosting` | **0.696** | **0.703** | 1.000 | 0.821 | 0.677 | 0.606 | 1.8 |
| `knn` | 0.620 | 0.624 | 0.967 | 0.786 | 0.581 | 0.515 | 0.1 |
| `svc` | 0.674 | 0.677 | 1.000 | 0.893 | 0.677 | 0.485 | 0.2 |
| `linear_svc` | 0.674 | 0.677 | 1.000 | 0.857 | 0.548 | 0.636 | 0.2 |

Macro-F1 leads and accuracy follows, for a reason the `dummy` row shows: predicting Low
every time is right 30% of the time and has a recall of 1.000 on Low and 0.000 on
everything else. Accuracy calls that a third of a model; macro-F1 calls it a sixth.
`adjacent` — right or one class out — is not a standard metric and is here because the
classes are ordered and no standard metric knows that.

### Against the baselines

`python -m stockout backtest`, five rolling-origin folds, MASE scaled by seasonal-naive
refitted on each training window, so *below 1 means beaten*:

| Model | WMAPE | MASE | |
|---|---|---|---|
| `svr` | 0.0818 | **0.417** | beats the baseline by 58.3% |
| `ridge` | 0.0849 | 0.430 | |
| `hist_gradient_boosting` | 0.0858 | 0.434 | |
| `moving_average` | 0.1442 | 0.732 | a trailing mean, and it beats seasonal-naive here |
| `seasonal_naive` | 0.2004 | 1.000 | the baseline, by definition |
| `naive_last` | 0.2073 | 1.043 | 4.3% worse |

The gap between `seasonal_naive` and `naive_last` is the value of knowing what day of the
week it is. That `moving_average` beats both is a property of this generator — its weekly
cycle is smooth enough that a trailing mean tracks it — and is exactly the sort of thing
that would not survive on real data. The gap at the top is worth less than it looks, for
the reason the flat ladder above gives.

## The leakage decomposition

`python -m stockout leakage` scores four training protocols against **one fixed future
window** that none of them trains on. The obvious experiment — score a random split, score
a time split, compare — cannot work, because those two differ in training size, test period
*and* leakage all at once ([ADR 0020](docs/decisions/0020-the-leakage-experiment-holds-the-future-fixed.md)).

`internal` is what the protocol would have told you; `future` is what that model then
scored; `optimism` is the difference, and it is the column the module exists to produce.

| arm | protocol | features | internal | future | optimism |
|---|---|---|---|---|---|
| honest | time-ordered | denylist enforced | 0.9013 | 0.8447 | 0.0566 |
| shuffled split | random | denylist enforced | 0.9131 | 0.8413 | 0.0718 |
| preprocessing leak | time-ordered | scaler and imputer fitted on all rows | 0.9014 | 0.8447 | 0.0567 |
| future feature | time-ordered | `customers` smuggled past the denylist | 0.9984 | 0.9988 | **−0.0004** |

**Two of the four find nothing on this data and it is reported anyway.** The shuffled
split's optimism exceeds the honest arm's by 0.015 R²; the preprocessing leak differs in
the fourth decimal. On a generator with a deterministic calendar there is no informative
neighbour for a shuffled split to see, which is a fact about the generator rather than
about the method.

**The fourth arm is the one that transfers.** `customers` is in `train.csv`, correlates
with `sales` at about 0.9, and nobody knows it six weeks ahead. It scores 0.998 on a
genuinely held-out window and is undeployable for a single day. Getting it into that arm
requires **renaming** it, because `FEATURE_DENYLIST` refuses it by name and cannot be
switched off — needing to lie to a guard in order to demonstrate what it prevents is the
strongest evidence available that the guard is load-bearing.

## The five questions

Committed before the analysis, each naming what would count as a *no*. Answered in
[`notebooks/01_explore.ipynb`](notebooks/01_explore.ipynb), written up with verdicts in
[docs/results.md](docs/results.md). **On synthetic data**, so these are demonstrations of
the machinery rather than claims about retail.

| | Question | On this data |
|---|---|---|
| Q1 | Does accuracy decay with horizon? | **No.** WMAPE 0.0849 → 0.0840 → 0.0835 → 0.0875 across 7, 14, 28, 42 days — flat, which was the stated failure condition |
| Q2 | Does the baseline beat a fitted model on quiet stores? | **No.** Every store is beaten, pooled (worst MASE 0.604) and per-store (worst 0.685) |
| Q3 | Is the error concentrated in a few days? | **Yes.** The worst 10% of days carry 29.2%; the five worst are all December |
| Q4 | Does the promotion lift persist, or reverse? | **It reverses, weakly.** +24.5% while running, −1.8% averaged over the following week |
| Q5 | Does the classifier beat binning the regressor? | **Yes.** 0.703 against 0.661 macro-F1, and it trades High recall for Low |

Three of five lost or came out flat. That is the point of writing them down first.

## The four traps this data sets

1. **A closed store sells zero.** About a seventh of Rossmann rows are closures. Every
   model predicts them perfectly, which flatters `mae` and `rmse` — they are per-row
   averages. `wmape` is immune, because a closed day cancels in both numerator and
   denominator. Both behaviours are asserted, because assuming either one is easy.
2. **`customers` is bait.** It predicts `sales` at about r = 0.9, it is in the training
   file, and nobody knows it six weeks ahead. It is on a denylist enforced by a test — and
   so are `demand_class` and `demand_class_code`, because two targets that encode each
   other make "not this task's target" too weak a rule.
3. **A lag shorter than the horizon is unobservable.** Forecasting six weeks out, you do
   not know yesterday's sales. `lag_1` in a horizon-42 model is the most common leak in
   public retail notebooks and the hardest to see, because nothing about the code looks
   wrong. Here it raises `LeakageError`.
4. **Refurbished stores vanish.** They stop appearing in the file entirely — absence, not
   a run of zeros — and produce a fake level shift on reopening. `validate.calendar_gaps`
   finds them. The synthetic generator reproduces this on purpose.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev,notebook]"
```

```bash
python -m stockout describe                            # rows, stores, nulls, calendar gaps
python -m stockout prepare                             # build the model frame once, cache it
python -m stockout backtest --model seasonal_naive     # the baseline
python -m stockout backtest --model naive_last         # what ignoring the weekday costs
python -m stockout backtest --model hist_gradient_boosting
python -m stockout compare                             # all twelve regressors
python -m stockout compare --task classification --models dummy logistic svc
python -m stockout leakage                             # the four-arm decomposition
python -m stockout tune --models ridge lasso elastic_net
python -m stockout train --regressor ridge             # fit both tasks, write one artifact
python -m stockout predict --store 1 --date 2015-01-02 --promo 1
python -m stockout synth --out data/mine.csv --stores 20 --days 1095
```

Every command that fits a model takes `--data`, `--stores` and `--horizon`, and builds its
frame through `dataset.prepare` — the one place the six preparation steps happen in the one
order that is correct. `compare`, `leakage` and `train` also take `--test-days` and
`--gap-days`, which defaults to the horizon rather than to zero: a holdout whose training
rows end the day before its test rows begin is scoring a one-day forecast however long the
horizon claims to be.

`predict` reads its horizon from the saved artifact rather than from a flag, and refuses a
date more than one horizon past the end of history — beyond that the model would have to
be fed its own output as an observation, and each prediction would inherit the last one's
error.

## Data

Everything above runs offline, on a committed synthetic sample. For the real thing:

```bash
python -m stockout fetch
python -m stockout compare --data data/raw/train.csv --stores data/raw/store.csv
```

`fetch` needs **two** things, and the second is the one people miss:

1. A Kaggle API token at `~/.kaggle/kaggle.json` — kaggle.com → Settings → API → *Create
   New Token*.
2. Acceptance of the [competition rules](https://www.kaggle.com/competitions/rossmann-store-sales/rules),
   signed in, on the web page. **Until that is done the API returns 403 for a token that
   is otherwise perfectly valid.** The downloader says so rather than surfacing a raw
   traceback.

The archive is never committed — see [ADR 0001](docs/decisions/0001-data-is-downloaded-not-committed.md).

## Layout

```
src/stockout/
├── data/schemas.py        every assumption about the file's shape, in one place
├── data/synth.py          deterministic sales generator — the suite runs on it
├── data/synth_stores.py   the metadata half, with the missing-value patterns that matter
├── data/validate.py       schema guard: dtypes, unique keys, open/sales invariant
├── dataset.py             the six preparation steps, in the one order that is correct
├── features/lags.py       leakage guard: shift before you roll, lag >= horizon
├── features/build.py      the model matrix, and the denylist
├── features/preprocess.py one ColumnTransformer, four branches, fitted inside the fold
├── features/store_features.py  store metadata crossed with the row's own calendar
├── targets.py             per-store demand terciles, fitted on the training window
├── split/rolling.py       the rolling-origin fold layout
├── split/strategies.py    the three protocols — the only module allowed to shuffle
├── models/spec.py         what the registry stores: a factory, plus how to read the score
├── models/registry.py     one list per task, so CLI, notebook and table cannot drift
├── models/adapter.py      frame-in/series-out over sklearn's fit(X, y)
├── models/{linear,trees,kernels,classifiers}.py   the twenty-one models
├── models/grids.py        the search spaces, as literals and nothing else
├── models/tuning.py       GridSearchCV over TimeSeriesSplit, never KFold
├── evaluate/backtest.py   the fold loop; a fresh model per fold
├── evaluate/metrics.py    WMAPE, MASE, RMSPE, R². Deliberately no MAPE
├── evaluate/classification.py  macro-F1, per-class recall, and adjacent accuracy
├── evaluate/comparison.py the results table, with rows fitted and seconds as columns
├── evaluate/leakage.py    four protocols, one fixed future
├── train.py               fit both tasks on everything, package the result
├── persistence.py         the artifact: model, thresholds, horizon, provenance
├── predict.py             rebuild the lags around one future day and read the last row
└── cli.py                 fetch | synth | describe | prepare | backtest | compare
                           | leakage | tune | train | predict

docs/questions.md       the five hypotheses, written first
docs/results.md         the answers, and what didn't work
docs/decisions/         thirteen current ADRs and seven superseded, each with its cost
notebooks/              the analysis; every cell is run by the test suite
data/                   gitignored, except two small synthetic samples
reports/                generated charts — regenerated, never committed
```

## Testing

```bash
ruff check .                                  # lint only; never `ruff format` (ADR 0004)
pyright                                       # type gate
pytest --cov --cov-fail-under=90              # 580 passing, 1 skipped, 98% covered
```

Unit tests never touch the network. A `conftest.py` autouse fixture replaces
`socket.connect`, so a test that reaches for a socket fails loudly instead of passing on a
laptop that happens to have credentials and failing in CI. Network work goes behind
`@pytest.mark.integration`, excluded by default.

`tests/test_no_random_splits.py` parses every module's AST and asserts that no shuffled
split executes anywhere in the package. It parses rather than greps because a text search
cannot tell code from the docstrings explaining why the code is absent — and would
therefore be defeated by deleting the explanation. `split/strategies.py` is the single
exemption, and it is skipped by name: a project cannot demonstrate what shuffling costs
while refusing to import the thing that shuffles.

`tests/test_notebook.py` execs every code cell of the notebook into one namespace and
checks that each question wrote its figure. It needs no kernel — an `.ipynb` is JSON — and
it exists because the previous notebook shipped importing six deleted modules and nothing
in the suite noticed.

`pyproject.toml` sets `filterwarnings = ["error"]`, so a `ConvergenceWarning` or an
unseen-category warning is a failed test rather than a line of yellow text. Several of the
values in `linear.py` and `classifiers.py` were raised until that stopped happening, which
is a worse reason than deriving them and a better one than taste.

## Known limits

- **The committed sample is synthetic.** It reproduces five structures that make the real
  problem hard — a weekly cycle, a promotion calendar, closures, refurbishment gaps and
  AR(1) residuals — and nothing else. Anything learned from it is a statement about a
  generator, and it is used to test machinery, never to support a finding. This is why
  three of the five questions come out flat or negative.
- **The ladder is flat, so the registry proves less than it demonstrates.** Twelve models
  land within 0.01 WMAPE of each other, and the honest reading is that this problem does
  not need eleven of them. Whether the ordering survives on Rossmann is unanswered.
- **The comparison rests on one holdout window.** 28 days, one draw. Every model in the
  table shares it, so the *ranking* is fair, but no confidence interval is attached to any
  number and none should be inferred. `backtest` is the multi-fold view and only runs one
  model at a time.
- **Three models are fitted on a subsample and are therefore judged unfairly.** `svr` and
  `svc` see at most 5,000 rows because an RBF fit is between quadratic and cubic in them —
  0.15s at 2,000 rows and 48s at 20,000. The cap is a printed column rather than a fix
  ([ADR 0015](docs/decisions/0015-a-subsample-is-reported-never-silent.md)), and on this
  four-store sample it never binds, so the mechanism is invisible in the numbers a fresh
  clone can produce.
- **The searched hyperparameters are not fed back.** `tune` prints what it found; the
  registry still holds its literals. Closing that loop automatically would make the
  registry a build artifact, so a reader has to run the search to learn that the literal is
  not the searched answer
  ([ADR 0017](docs/decisions/0017-every-hyperparameter-is-searched-not-defaulted.md)).
- **Pooled, but only by accident.** The baselines are strictly per store. Every
  scikit-learn model is one global fit with `store` as an integer feature, which pools by
  default and shares nothing deliberately — no hierarchy, no per-store effects, no
  borrowing toward a group mean. Short histories are served worst either way.
- **`store` reaches the linear models as an unscaled integer.** A linear fit will give it a
  coefficient as though store number were a quantity. It is near zero on this data because
  store id does not trend with sales, and nothing enforces that or would detect it changing
  ([ADR 0019](docs/decisions/0019-store-is-an-ordinal-id-not-1115-one-hot-columns.md)).
- **The demand classes are not a business's classes.** No retailer defines a busy day as
  the top third of that shop's own history. A real definition would come from a planner and
  would probably not be a tercile
  ([ADR 0016](docs/decisions/0016-demand-classes-are-per-store-terciles-fitted-on-the-training-window.md)).
- **A store with no training rows takes the pooled cut points**, which is the global
  threshold this project argues against, applied to a minority of rows. Roughly a sixth of
  Rossmann's stores vanish for a refurbishment quarter, so it is not a hypothetical branch.
- **There is no decision layer any more.** The project scores forecasts on accuracy, and
  accuracy is not what a replenishment decision is graded on. That claim was the strongest
  thing this repository had and it went with the inventory simulator
  ([ADR 0014](docs/decisions/0014-a-scikit-learn-comparison-not-an-inventory-system.md)).
- **`predict` answers one store-day at a time.** It rebuilds features over that store's own
  history per call, which is right and is not batched; a thousand stores is a thousand
  rebuilds.
- **Loading an artifact runs code from it.** `joblib.load` executes what it reads, so
  `persistence.load` refuses any path outside the artifact directory. That is a guard
  against a careless caller, not a sandbox.

## Documentation

- [docs/questions.md](docs/questions.md) — the five hypotheses, committed before analysis
- [docs/results.md](docs/results.md) — the answers, and what didn't work
- [docs/architecture.md](docs/architecture.md) — flow, the three guards, module map
- [docs/data-dictionary.md](docs/data-dictionary.md) — columns, traps, future-known vs observed
- [docs/glossary.md](docs/glossary.md) — the forecasting and modelling terms used here
- [docs/decisions/](docs/decisions/) — the ADRs, each with its cost
- [MODEL_CARD.md](MODEL_CARD.md) — what the models are, and are not, for

## Licence

MIT — see [LICENSE](LICENSE). The Rossmann dataset is not redistributed here and remains
subject to the competition's own terms.
