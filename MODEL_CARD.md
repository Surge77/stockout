# Model card — stockout

**Status: twenty-four models, evaluated on synthetic data only.** Every number below comes
from a generator, not from retail, and is stated that way wherever it appears.

## What exists

Three **baselines**, fitted per store on trading days only. They exist so that every other
number has something to be compared against: a fitted model's R² of 0.84 means nothing
until you know what a rule a shopkeeper could apply in their head scores.

| Model | Rule | Parameters |
|---|---|---|
| `naive_last` | Most recent trading-day figure | none |
| `seasonal_naive` | Most recent same store, same weekday | `season_length = 7` |
| `moving_average` | Mean of the last `window` trading days | `window = 28` |

Twelve **regressors** predicting `sales`, and nine **classifiers** predicting a
Low/Medium/High demand class, all behind one shared preprocessor.

| Family | Regression | Classification | The question it answers |
|---|---|---|---|
| floor | `dummy` | `dummy` | what does knowing nothing score? R² is *defined* as the improvement on the mean; three balanced classes put accuracy at 33% |
| linear | `linear`, `ridge`, `lasso`, `elastic_net`, `polynomial` | `logistic` | is a weighted sum enough, and what do the three penalties change? |
| trees | `decision_tree`, `bagging`, `random_forest`, `hist_gradient_boosting` | the same four | do interactions matter, and what is bagging-plus-feature-subsampling worth over bagging? |
| kernel & instance | `knn`, `svr` | `knn`, `svc`, `linear_svc` | does a non-parametric boundary beat a parametric one, and what does the kernel trick cost? |

`bagging` and `random_forest` differ by exactly one idea — a random feature subset at each
split — so the gap between them isolates that idea. `linear_svc` exists so the cost of the
kernel trick is visible against a model solving the same problem in linear time.

**Preprocessing is shared and fitted inside each fold.** Numeric columns are median-imputed
(`competition_distance` is heavily right-skewed) and standardised; four low-cardinality
categoricals are one-hot encoded with `handle_unknown="ignore"`; `promo_interval` goes
through Tf-idf over a fixed month vocabulary; `store` passes through as an integer rather
than becoming 1115 columns and 3.77 GB
([ADR 0019](docs/decisions/0019-store-is-an-ordinal-id-not-1115-one-hot-columns.md)).
`drop_first` is applied per model — the dummy-variable trap is a linear-model problem and a
tree loses a usable split by it.

**Features** are calendar covariates knowable in advance, three date-relative store
features derived from the metadata join, plus lags and rolling statistics that are all at
least one horizon old. `customers`, `demand_class` and `demand_class_code` are excluded by
a denylist enforced by a test.

Every regressor predicts zero when the trading calendar says the store is shut, and every
classifier abstains rather than labelling a closed day. That calendar is future-known — it
comes from a planning system, not an observation — so using it is not leakage.

## The classification target is a definition, not a given

"High demand" has no meaning until somebody fixes the cut points. Here they are the 1/3 and
2/3 quantiles of **each store's own trading-day sales**, fitted on the **training window
only**, with a pooled pair as fallback for a store the training window never saw
([ADR 0016](docs/decisions/0016-demand-classes-are-per-store-terciles-fitted-on-the-training-window.md)).

The alternative is measured rather than argued against. With one global pair of cut points,
`logistic` scores **0.810** macro-F1 instead of 0.691 — a larger gap than separates any two
models in the entire regression table. It has not become a better model; it has been handed
a shortcut, because global thresholds put every day of a busy shop in *High* and the
classifier can read store identity straight off the `store` column.

## Intended use

Comparing model families on a problem where the usual validation protocol is a bug, and
exercising the pipeline end to end. **Not** for making real replenishment decisions — see
the first limitation below.

## Training data

The committed sample is **synthetic** (`stockout.data.synth` and
`stockout.data.synth_stores`, 4 stores × 730 days, seed 7). It deliberately reproduces five
structures — weekly seasonality, a promotion calendar, Sunday closures, a refurbishment gap
where rows vanish entirely, and AR(1) residuals — and nothing else.

Real runs use Rossmann Store Sales (1115 German drugstores, 2013-01-01 to 2015-07-31),
downloaded per machine and never redistributed.

## Evaluation

Two protocols, because two questions need different shapes of evidence.

**`compare`** — one held-out window, every model, so the only thing varying between two
rows is the estimator. 28 days with a 7-day gap in front of it, scored on trading rows
only, 2,279 training rows and 92 scored. This is the table that decides *which model*.

**`backtest`** — one model across five rolling-origin folds on a 365-day minimum expanding
training window. This is the only path producing MASE, because MASE needs a baseline
refitted on each training window. Closed days are excluded from both. Metrics: WMAPE
(primary), MASE against seasonal-naive, RMSPE (Kaggle's), R² (reported because it is
expected, and it is not what decides anything). MAPE is deliberately absent —
[ADR 0005](docs/decisions/0005-wmape-and-mase-not-mape.md).

### Regression, on the held-out window

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

**The ladder is flat and that is the result.** A straight line scores WMAPE 0.0899; so does
the best model in the table. On a generator whose promotion calendar and weekday pattern
are deterministic, a linear model on the right features is the correct answer.

### Classification, on the same window

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

Macro-F1 leads and accuracy follows. The `dummy` row shows why: predicting Low every time
is right 30% of the time with recall 1.000 on Low and 0.000 on everything else. Accuracy
calls that a third of a model; macro-F1 calls it a sixth. `adjacent` — right or one class
out — is not a standard metric and is here because the classes are ordered and no standard
metric knows that.

### Against the baselines, across five folds

| Model | WMAPE | MASE |
|---|---|---|
| `svr` | 0.0818 | **0.417** |
| `ridge` | 0.0849 | 0.430 |
| `hist_gradient_boosting` | 0.0858 | 0.434 |
| `moving_average` | 0.1442 | 0.732 |
| `seasonal_naive` | 0.2004 | 1.000 |
| `naive_last` | 0.2073 | 1.043 |

That `moving_average` beats `seasonal_naive` is a property of this generator — its weekly
cycle is smooth enough for a trailing mean to track — and is the kind of ordering that
would not survive real data.

**These are synthetic-data figures and are not evidence about retail forecasting.**
Reproduce with `python -m stockout compare` and `python -m stockout backtest --model ridge`.

### What is deployed

`python -m stockout train` fits one regressor and one classifier — both
`hist_gradient_boosting` by default — on **every** row, and writes them with the demand
thresholds, the horizon and the provenance into one artifact. On the committed sample that
is 2,395 training rows and 35 feature columns.

The scores stored alongside are **not** from that fit. They come from the held-out
comparison, because a score computed on the rows a model was fitted on is not a score.

## Limitations

- **There is no decision layer.** Every model here is scored on accuracy, and accuracy is
  not what a replenishment decision is graded on: low WMAPE and low cost are different
  orderings. An earlier version of this project priced the decision with a newsvendor
  simulator and it was deleted
  ([ADR 0014](docs/decisions/0014-a-scikit-learn-comparison-not-an-inventory-system.md)).
  Nothing replaces it, so a model that wins the table here is not thereby the model that
  should be stocked to.
- **The ladder is flat, so the comparison demonstrates more than it proves.** Twelve models
  land within 0.01 WMAPE of each other. The honest reading is that this problem does not
  need eleven of them, and whether the ordering survives on Rossmann is unanswered.
- **One holdout window, one draw.** Every model shares it, so the *ranking* is fair, but no
  confidence interval is attached to any number and none should be inferred. `backtest` is
  the multi-fold view and runs one model at a time.
- **Three models are fitted on a subsample and are judged unfairly for it.** An RBF fit is
  between quadratic and cubic in the training rows — 0.15s at 2,000 and 48s at 20,000 — so
  `svr` and `svc` see at most 5,000, and `knn` and `linear_svc` at most 50,000. The cap is
  a printed column rather than a fix
  ([ADR 0015](docs/decisions/0015-a-subsample-is-reported-never-silent.md)), and a fair
  comparison would cap everything, which would discard the main finding available: the
  models that can use all the data win because they can use all the data.
- **The cap never binds on the committed sample**, which has 2,395 training rows, so the
  mechanism is invisible in the numbers a fresh clone can produce.
- **The searched hyperparameters are not fed back into the registry.** `tune` prints what
  it found; `linear.py` still says `alpha=1.0`. On this sample the search picks a value
  0.002 R² *worse* on the held-out window than the default it replaced, so what it buys
  here is provenance rather than accuracy
  ([ADR 0017](docs/decisions/0017-every-hyperparameter-is-searched-not-defaulted.md)).
- **Pooled by default, structured not at all.** The baselines fit per store. Every
  scikit-learn model is a single global fit with `store` as an integer feature, so it pools
  incidentally rather than by design — no hierarchy, no per-store effects, no shrinkage
  toward a group mean. Stores with short histories are served badly either way.
- **`store` reaches the linear models as an unscaled integer**, and a linear fit will give
  it a coefficient as though store number were a quantity. It is near zero on this data
  because store id does not trend with sales; nothing enforces that or would detect it
  changing.
- **A store with no training rows takes the pooled cut points**, which is the global
  threshold this card argues against, applied to a minority of rows. Roughly a sixth of
  Rossmann's stores vanish for a refurbishment quarter, so it is not hypothetical.
- **The demand classes are not a business's classes.** No retailer defines a busy day as
  the top third of that shop's own history, and a real definition would come from a planner.
- **The baselines ignore promotions** despite `promo` being available and future-known, and
  predict one number per store (or store-weekday) for the whole horizon, so they cannot
  represent a trend or an approaching event. Every fitted model does use them.
- **`predict` answers one store-day per call**, rebuilding features over that store's own
  history each time, and refuses a date more than one horizon past the end of history.
  Beyond that the model would be fed its own output as an observation.
- **Loading an artifact runs code from it.** `joblib.load` executes what it reads, so
  `persistence.load` refuses any path outside the artifact directory. That is a guard
  against a careless caller, not a sandbox.
- **Synthetic evaluation only**, so far.

## Ethical and practical considerations

Store-level revenue aggregates; no personal data, and no individual is identifiable. The
`customers` column is a daily count, not a record of people, and is excluded from the
feature set anyway.

The realistic harm from a deployed version of this is commercial: an under-forecast empties
shelves, an over-forecast ties up cash and, for perishables, creates waste. That asymmetry
is real and **this project no longer models it**. Squared error treats a currency unit of
over-forecast and a currency unit of under-forecast identically, and every regression model
here is fitted to squared error. Anyone deploying this should assume the model is optimising
the wrong thing for their business and measure the gap before trusting it.

## What would change this card

Running any of it on the real Rossmann file. Every figure above is a statement about
`stockout.data.synth`.

The five questions in [docs/questions.md](docs/questions.md) are now **answered** on
synthetic data — three of the five lost or came out flat, which is what committing them
before the analysis is for — and answering them on real data is one `python -m stockout
fetch` and two changed lines at the top of the notebook.

Beyond that, the honest list of what this card cannot currently say: whether the model
ordering is a fact about the data or about the generator, what any of these forecasts costs
to stock to, and whether a per-store or hierarchical model beats the pooled fit on the 1115
stores where that question becomes answerable.
