# stockout — Retail Sales and Product Demand Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Checked with pyright](https://img.shields.io/badge/types-pyright-blue)](https://github.com/microsoft/pyright)

**This project predicts retail sales and product demand.** Give it a store and a day and it
answers two questions: how much will that store sell, and is that a Low, Medium or High
demand day *for that store*. Named after the failure it exists to prevent.

The domain is fixed. This package works on retail store-day data and never on unrelated
tabular datasets.

| | |
|---|---|
| **Regression target** | store-day `sales` |
| **Classification target** | demand class — Low / Medium / High, from that store's own terciles |
| **Input** | store attributes, calendar position, promotions, holidays |
| **Data** | a committed synthetic sample, and the real Rossmann competition data |
| **Interfaces** | a CLI, and a web app with a forecast panel, a what-if panel and an admin panel |

```bash
pip install -e ".[web]"

python -m stockout datasets                                  # what is available
python -m stockout describe                                  # rows, dtypes, demand balance
python -m stockout models                                    # what --regressor accepts
python -m stockout compare --task regression                 # the seven regressors
python -m stockout compare --task classification             # the six classifiers
python -m stockout leakage                                   # what a shuffled split adds
python -m stockout train                                     # fit both, save one artefact
python -m stockout predict --store 12 --date 2015-03-06 --promo

uvicorn stockout_web.main:app                                # the panels, on :8000
```

## The three panels

**User panel** (`/`) — pick a store, a date and whether a promotion runs. It answers with
predicted sales *and* the demand class side by side, because they come from two different
models that can disagree, plus that store's own cut points. Where the data has a true
figure for that store-day it is shown **first**, with the error beside it — see
[the date policy](#the-date-policy) for why. An expandable section lists what every
registered regressor predicted for the same row:

```
Store 12 — Friday 06 March 2015, promotion on

  Predicted sales   7,409          Demand class   High
  gradient boosting                random forest

  This store's cut points: Low <4,297 · Medium 4,297–5,174 · High >5,174

  ▾ What every model predicted for this row (7)
      polynomial          8,663
      gradient boosting   7,409   [deployed]
      ridge               7,396
      linear              7,396
      lasso               7,381
      voting              7,026
      random forest       6,271
```

They span 6,271 to 8,663. That spread is the honest width of the estimate, and hiding it
behind one number would imply a certainty nobody has.

**What-if panel** (`/what-if`) — no store id at all. Describe a shop by its type,
assortment, distance to the nearest competitor and whether it runs the continuing
promotion, and get an estimate for one day, the way a house-price model works from floor
area and location rather than from an address:

```
A shop like that, on Monday 15 June 2015

  Estimated sales   15,785        Demand class   High
  cold-start model, no store id   against all stores, not one

  What each attribute is worth here
      Assortment            +3,565
      Competitor distance   +3,372
      Promotion             +3,236
      Store type            +1,203
      Promo2                  +519
```

This uses a **second regressor**, fitted without `entity_code` and without the competitor
and promotion *ages* — nobody describing a shop that does not exist can supply when its
hypothetical competitor opened, and leaving those in sent two very different shops down
the same branch, within 4% of each other. It is scored leave-one-store-out rather than on
the ordinary hold-out, because that is the question it answers.

The contributions are one-at-a-time counterfactuals, not an additive decomposition. They
do not sum to the total and are not meant to; a tree ensemble is not additive.

**Admin panel** (`/admin`) — model cards with a bar per model, the deployed pair's
provenance and held-out scores, retrain-and-redeploy with any pair from the registry, the
full comparison table for both tasks, and account management.

A regressor has no accuracy and a classifier has no r², so each card names the metric its
own task actually produces. Nothing on the page is a number that cannot exist.

**Accounts** — open registration; the **first account created becomes the admin** and every
one after it is a normal user, so no default credentials ship in the repository. The role is
re-read from the database on every request rather than trusted from the cookie, so demoting
or deactivating somebody takes effect on their very next request. The last active admin can
be neither demoted nor deactivated.

## The data

**`data/sample_sales.csv` + `data/sample_stores.csv`** — 54,660 store-days across 50 stores
and three years, Rossmann-shaped and generated deterministically. Committed, so a fresh
clone runs offline. Regenerate or resize with `python scripts/make_sample.py 50 1095`.

**Rossmann** — the real competition data, 1,115 stores over 942 days. Kaggle forbids
redistributing it, so it is downloaded per machine:

```bash
python -m stockout fetch          # needs ~/.kaggle/kaggle.json
python -m stockout compare --dataset rossmann
```

You must accept the [competition rules](https://www.kaggle.com/competitions/rossmann-store-sales/rules)
while signed in first, or the API answers with a bare 403. Everything works on the
synthetic sample until you do.

Both sources go through an adapter into one canonical shape — `entity, date, sales, open,
promo, state_holiday, school_holiday` — so the registry, the comparison table and the panels
are written once. Adding another retail source means writing one loader in
`src/stockout/data/adapters.py`.

## Features, and what this can honestly claim

Every feature is knowable **before** the day happens:

| Source | Features |
|---|---|
| the date | `day_of_week`, `month`, `day_of_month`, `week_of_year`, `is_weekend` |
| the row | `promo`, `state_holiday`, `school_holiday` |
| store metadata | `store_type`, `assortment`, `competition_distance`, `competition_open_age_months`, `promo2`, `promo2_age_weeks`, `is_promo2_month` |
| identity | the store's integer id — never one-hot, since real Rossmann has 1,115 of them |

**There are no lag features and no rolling means.** Nothing is derived by shifting the sales
column, so the textbook leak — a model shown yesterday's sales while predicting today's —
cannot happen here.

That is not the same as a shuffled split being safe, and the difference was worth
measuring rather than assuming. `entity_code` together with `month`, `day_of_month` and
`day_of_week` very nearly identifies a single row, so a model with enough capacity can
isolate a test row inside a neighbourhood of the days around it — and adjacent days are
correlated by construction. Shuffling hands those neighbours over. It is worth up to
**+0.088 r²** here, which is 15% of the honest score for the worst-affected model. The
hold-out is chronological for that reason; `python -m stockout leakage` prints the table,
and [Results](#results) reproduces it.

What the feature set costs, stated rather than hidden: this answers *what sales should we
expect at this store on this kind of day*, **not** *forecast the next six weeks*.

**Closed days are dropped.** A shut store sells exactly zero and every model predicts zero,
so keeping those rows hands every score a block of free correct answers — 7,937 of the
54,660 rows here, about a seventh.

**Demand classes are per-store terciles fitted on the training rows alone.** Global cut
points would make a big store "High" on 94% of days, so the classifier would learn store
identity, score ~95% and know nothing. Per-store terciles ask the question that matters:
*is today busy for this shop?* The cut points travel in the artefact, because a predicted
number cannot be turned into a class without them.

## The models

Thirteen, each wrapped as `Pipeline([("Scaler", StandardScaler()), ("Model", estimator)])`.
Ten come from the course notebooks in `ML/Resource Files`; the other three are named
exceptions and are flagged in every table this package prints.

| Task | Name | Estimator | Taught |
|---|---|---|---|
| regression | `linear` | `LinearRegression()` | yes |
| regression | `ridge` | `Ridge(alpha=1)` | yes |
| regression | `lasso` | `Lasso(alpha=100)` | yes |
| regression | `polynomial` | `PolynomialFeatures(2)` → scaler → `LinearRegression()` | yes |
| regression | `random_forest` | `RandomForestRegressor()` | **no** |
| regression | `gradient_boosting` | `GradientBoostingRegressor()` | **no** |
| regression | `voting` | `VotingRegressor([linear, random_forest, gradient_boosting])` | **no** |
| classification | `logistic` | `LogisticRegression(max_iter=1000)` | yes |
| classification | `knn` | `KNeighborsClassifier()` | yes |
| classification | `decision_tree` | `DecisionTreeClassifier()` | yes |
| classification | `random_forest` | `RandomForestClassifier()` | yes |
| classification | `bagging` | `BaggingClassifier()` | yes |
| classification | `svc` | `SVC()` | yes |

Categoricals go through `LabelEncoder` with a **fixed** vocabulary, so a store type the
training rows never contained cannot raise inside a web request. Label encoding implies an
ordering that does not exist; trees cope with that and the linear models less so.

The guard that keeps this honest:

```bash
python scripts/check_taught.py            # exits 0
python scripts/check_taught.py --strict   # exits 1, naming exactly those three
```

It parses the AST rather than grepping — a `grep` cannot see a parenthesised multi-line
import, and one silently passed while hiding three names before this was written.

## Why a tree alone cannot forecast

Gradient boosting and random forests **cannot extrapolate**. A tree splits on a threshold
and returns a leaf value; asked about a point past every threshold it ever built, it lands
in the same leaf as the largest value it saw and answers with that. Handing such a model a
time index does not help, which is the trap — the feature looks like it should work and
buys nothing outside the training range.

Rossmann is not flat. Over 2013-01-01 to 2015-07-31 the chain grows about **5.1% a year**,
the median shop 3.7%, and 68% of shops grow faster than 2% while 10% shrink faster than 2%.
A model blind to that is systematically ~5% low a year out.

So the level goes somewhere a tree is not — a straight line per shop, fitted on the
training rows alone, subtracted before the tree sees the target and added back after:

```
sales  ≈  (aᵢ + bᵢ·t)   +   tree(calendar, promotions, shop attributes)
          └ extrapolates ┘   └────── does not need to ──────┘
```

What that is worth, same estimator and same rows, only the target differs:

| dataset | raw sales | trend removed | change |
|---|---|---|---|
| rossmann (grows 5.1%/yr) | 0.3401 | **0.8296** | **+0.490** |
| synthetic (no chain trend) | 0.9017 | 0.9046 | +0.003 |

The second row is the one that makes it safe as a default: where there is no trend it
correctly finds none — median fitted growth on synthetic is +0.05%/yr — and does nothing.
`--no-detrend` reproduces the left column.

It is **not a lag**, so nothing in [the leakage argument](#what-the-taught-split-was-worth)
changes: `t` for any future day is knowable today.

**It repaired the classifier too.** Demand bands cut on raw sales drift as shops grow — on
Rossmann the held-out period reaches 40.6% High against 33.3% in training, purely because
the chain gained ground. Cut on the residual they stay near a third, and "High" goes on
meaning *busy for this shop, now* rather than *big in 2013 terms*.

| bands cut on | training | held out |
|---|---|---|
| raw sales | 33.4 / 33.3 / 33.3 | 25.9 / 33.6 / **40.6** |
| trend removed | 33.3 / 33.3 / 33.4 | 36.9 / 32.6 / 30.5 |

**What a straight line cannot do**, said here rather than discovered later: it assumes the
trend continues, so a competitor opening or a refurbishment breaks it; it cannot represent
a trend that *changes*; and far enough out it is arithmetic rather than a forecast, which
is what `trend.MAX_CONFIDENT_DAYS` exists to say out loud.

## The date policy

Every feature is built from the calendar or from store metadata. Nothing is a lag, so
scoring a day needs no history *of that day* — which makes "which dates may be asked
about" a policy question, not a technical one. `stockout/window.py` holds the policy, and
both the CLI and the panel go through it. They did not always: the guard used to live in
the web layer alone, and `stockout predict --date 2030-01-01` answered without comment.

Three regions.

| region | answer | why |
|---|---|---|
| before `2013-01-01` | refused | nothing behind it in either direction |
| `2013-01-01`–`2015-12-31` | answered, **with the actual** | the true figure is in the data |
| after `2015-12-31` | answered, marked a forecast | nothing to check against, nothing measured |

The middle row is the honest reply to the obvious objection. Inside the window a prediction
*is* redundant — `SELECT sales WHERE store = ? AND date = ?` beats every model in the
registry. So the panel stops pretending otherwise and shows both:

```
Store 12 — Friday 06 March 2015, promotion on

  What it actually took   8,255      recorded in the data
                                     +1,157 (+14.0%) prediction error
  Predicted sales         9,412      gradient boosting
```

That is the only place on the site where a visitor watches the model be checked rather than
trusted. Outside the window there is nothing to check against, and the page says so.

**Forward is allowed and backward is not**, which needs one piece of care. The calendar
features are *cyclic* — `day_of_week`, `month`, `week_of_year` for a day in 2030 take
values the fit saw thousands of times, so they extrapolate for free. The store ages are
*monotone*: `competition_open_age_months` grows without bound, and on a date years out it
arrives beyond every split point the trees were built from, on a feature carrying about a
sixth of the deployed model's importance. `features.build` caps the ages at the last day of
data and leaves the calendar alone, so the row stays in distribution on the feature that
would otherwise leave it.

One consequence worth stating plainly. With the ages capped, *06 March 2026* and *06 March
2015* are the same kind of day — both Fridays in week 10 — and the model returns the same
number for both. That is correct and it is also the limit: this forecasts a **seasonal
profile**, not a trend. It cannot tell you the chain grew 4% next year, because nothing in
its inputs could.

## Results

> **The tables in this section are synthetic and now understate the project.** They predate
> the switch to real data and have not been regenerated — `compare --dataset rossmann` for
> both tasks takes several minutes each. The real figures measured so far are below and in
> [the trend section](#why-a-tree-alone-cannot-forecast).

### On the real competition data

```
$ python -m stockout compare --dataset rossmann --task regression --no-detrend
rossmann: 844,392 open store-days | 1,115 stores | 2013-01-01 to 2015-07-31
regression | 7 models | 173,970 held-out rows | raw sales, no trend term

| model             | r2     | fit_seconds | n_fit  | taught |
|-------------------|--------|-------------|--------|--------|
| linear            | 0.1821 | 0.622       | 670422 | yes    |
| ridge             | 0.1821 | 0.493       | 670422 | yes    |
| lasso             | 0.1684 | 0.525       | 670422 | yes    |
| polynomial        | 0.2102 | 15.581      | 670422 | yes    |
| random_forest     | 0.8240 | 343.379     | 670422 | no     |
| gradient_boosting | 0.3401 | 67.727      | 670422 | no     |
| voting            | 0.5816 | 468.061     | 670422 | no     |
```

Two rows are worth pausing on, because together they are the argument for the trend term.

`random_forest` wins at 0.8240 while `gradient_boosting` manages 0.3401 — an odd gap, since
boosting normally matches or beats a forest. The cause is depth: a forest of 100 deep trees
can memorise each shop's level through `entity_code`, and sklearn's boosting default of
`max_depth=3` cannot isolate 1,115 shops. The forest is not modelling demand better; it is
remembering which shop it is looking at.

That distinction is invisible in this table and decisive in use. De-trended boosting scores
**0.8296** — statistically the same as the forest's 0.8240 — but the forest's score comes
from memorising levels *inside* the training window, so it still flatlines on any date past
the data. Two models, one number, and only one of them can forecast.

`svc` is absent because it is a classifier; on the classification table it is fitted on a
capped 50,000-row sample, which the `n_fit` column reports. Kernel SVC is quadratic in rows,
so 670,422 of them is hours of work for one cell.

### On the synthetic fixture


`compare` fits every registered model on the same training rows and scores it on the same
test rows, so the only thing that differs between two rows of a table is the estimator.

**The hold-out is chronological.** The models are fitted on the earlier part of the
calendar and scored on a later part they have never seen — every training day strictly
before every test day. The taught `train_test_split` is one flag away (`--split shuffled`),
and [what it is worth](#what-the-taught-split-was-worth) is measured below.

```
$ python -m stockout compare --task regression
synthetic: 46,723 open store-days | 50 stores | 2013-01-01 to 2015-12-31
regression | 7 models | 9,350 held-out rows

| model             | r2     | mse            | fit_seconds | taught |
|-------------------|--------|----------------|-------------|--------|
| linear            | 0.6033 | 1,942,007.8884 | 0.0390      | yes    |
| ridge             | 0.6033 | 1,941,997.1603 | 0.0180      | yes    |
| lasso             | 0.5951 | 1,982,210.1469 | 0.1110      | yes    |
| polynomial        | 0.8246 | 858,679.7087   | 0.3730      | yes    |
| random_forest     | 0.8822 | 576,591.5147   | 14.7600     | no     |
| gradient_boosting | 0.9017 | 481,249.1160   | 4.0490      | no     |
| voting            | 0.8728 | 622,736.5830   | 18.8590     | no     |
```

A straight line reaches r² 0.60 because a store's attributes move its sales in a mostly
monotone way — a wider range sells more, a nearer competitor hurts. The quadratic
expansion adds 0.22 by picking up the interactions a line cannot represent (a promotion
on a Monday is not a promotion plus a Monday), and gradient boosting adds another 0.08.

```
$ python -m stockout compare --task classification
floor: 44.5% (always guess the commonest class)

| model         | accuracy | fit_seconds | taught |
|---------------|----------|-------------|--------|
| logistic      | 0.6178   | 0.1470      | yes    |
| knn           | 0.6782   | 0.0400      | yes    |
| decision_tree | 0.6999   | 0.2370      | yes    |
| random_forest | 0.7235   | 4.2780      | yes    |
| bagging       | 0.7273   | 1.3380      | yes    |
| svc           | 0.7650   | 65.9390     | yes    |
```

The floor is 44.5%, not the third terciles would suggest, and the gap is the chronological
split showing its work. Per-store terciles split the *training* period into near-equal
thirds; the held-out period is later, sales have drifted, and the same cut points no
longer divide it evenly. Every model clears the higher bar.

### What a store's attributes are worth

The headline result of the package, and the one it was rebuilt around. Hold an entire
store out of training, then predict it:

```
$ python -m stockout train
...
cold-start (leave-one-store-out) r2 = 0.703
```

That number was **-0.696** before `data/synth.py` changed. A store's trading level used
to be `rng.integers(3_500, 9_000)` — drawn at random, in a different function from the one
that generated its metadata — so nothing tied an attribute to the level it was supposed to
explain. Measured on the old data: `competition_distance` correlated with mean sales at
r = -0.13, assortment moved the mean by 8%, and `entity_code` **alone** reached r² 0.49.
The only way to know a store's level was to memorise which store it was.

| | attributes are decoration | attributes cause sales |
|---|---|---|
| `competition_distance` vs mean sales | r = −0.13 | **r = +0.48** |
| assortment, spread of group means | 15.1% | **38.3%** |
| r² without `entity_code` | 0.7927 | **0.8993** |
| r² from `entity_code` alone | 0.4899 | 0.4361 |
| **held-out store** | **−0.696** | **+0.703** |

The middle row is the one that matters: dropping store identity now costs 0.002 r², so the
model is reading attributes rather than remembering ids. That is what makes
[the what-if page](#the-three-panels) possible.

### What the taught split was worth

Same data, same seed, same models — only the division differs, so the gap is the part of a
score that comes from how the rows were split.

```
$ python -m stockout leakage --task regression

linear             r2  shuffled 0.6181   chronological 0.6033   optimism +0.0148
polynomial         r2  shuffled 0.8641   chronological 0.8246   optimism +0.0395
random_forest      r2  shuffled 0.9171   chronological 0.8822   optimism +0.0349
gradient_boosting  r2  shuffled 0.9137   chronological 0.9017   optimism +0.0120

largest optimism: polynomial at +0.0395 (+4.8% of its honest score)
```

Nothing in `features.py` is a lag, so this is not the textbook leak. `entity_code` with
`month`, `day_of_month` and `day_of_week` very nearly identifies one row, which lets a
model isolate a test row inside a neighbourhood of the days around it, and the generator's
residual is an AR(1) correlated 0.7 at one day. Shuffling hands those neighbours over.

Classification shows the same mechanism from the other side:

```
$ python -m stockout leakage --task classification

logistic       accuracy  shuffled 0.6022  chronological 0.6178  optimism -0.0155
random_forest  accuracy  shuffled 0.7546  chronological 0.7235  optimism +0.0311
bagging        accuracy  shuffled 0.7373  chronological 0.7273  optimism +0.0100
svc            accuracy  shuffled 0.7694  chronological 0.7650  optimism +0.0044

largest optimism: random_forest at +0.0311 (+4.3% of its honest score)
```

Only the bagged ensembles gain. `logistic` and `decision_tree` do slightly *better* on the
honest split, which is what the explanation predicts: exploiting a neighbourhood takes the
capacity to isolate one, and a linear boundary does not have it. An effect appearing
uniformly across every model would be a bug in the experiment rather than a finding.

This is what the AR(1) residual in `data/synth.py` is for. With independent noise both arms
score the same and the measurement cannot tell a good split from a bad one.

## Layout

```
src/stockout/            the library — pure, no I/O beyond reading its own data
  cli.py commands.py     the subcommands
  config.py errors.py
  data/
    schemas.py           the canonical column names
    adapters.py          synthetic | rossmann -> one shape
    download.py          the Kaggle fetch
    synth.py synth_stores.py loaders.py
  datasets.py            load, build features, hand back a model-ready frame
  features.py            calendar + store features + the LabelEncoder vocabulary
  targets.py             per-store terciles, fitted on training rows
  hypothetical.py        a described store, with no id, and what each attribute is worth
  models.py registry.py  the thirteen estimators
  evaluate.py            r2 / mse / accuracy, the compare loop, both splits
  leakage.py             prices the taught split against the chronological one
  window.py              which dates may be asked about, and what an answer means
  train.py               fits BOTH models, saves one artefact
  predict.py             one store-day -> sales + demand class
  persistence.py         artefact v3
  plots.py

src/stockout_web/        the app — nothing but I/O
  main.py service.py     startup, and the one deployed ModelService
  auth.py db.py users.py bcrypt, SQLite, roles
  routers/               auth | forecast | admin
  templates/ static/

tests/test_auth.py       auth and access control
tests/test_split.py      the chronological hold-out keeps its wall
tests/test_hypothetical.py  attributes carry signal; a held-out store is predictable
tests/web/               the fitted-date window, and the error page
scripts/                 make_sample.py, check_taught.py
```

## Development

```bash
pip install -e ".[web,dev,notebook]"

ruff check .                     # lint only — never `ruff format`
pyright                          # must be 0 errors
python scripts/check_taught.py   # no undocumented sklearn import
pytest                           # 10 auth tests
```

There is no test suite beyond auth. Everywhere else a bug is a wrong number; there it is a
way in. The rest is verified by running the commands above and executing the notebook.

## Licence

MIT — see [LICENSE](LICENSE).
