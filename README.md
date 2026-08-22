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
| **Interfaces** | a CLI, and a web app with a user panel and an admin panel |

```bash
pip install -e ".[web]"

python -m stockout datasets                                  # what is available
python -m stockout describe                                  # rows, dtypes, demand balance
python -m stockout models                                    # what --regressor accepts
python -m stockout compare --task regression                 # the seven regressors
python -m stockout compare --task classification             # the six classifiers
python -m stockout train                                     # fit both, save one artefact
python -m stockout predict --store 12 --date 2015-03-06 --promo

uvicorn stockout_web.main:app                                # the panels, on :8000
```

## The two panels

**User panel** (`/`) — pick a store, a date and whether a promotion runs. It answers with
predicted sales *and* the demand class side by side, because they come from two different
models that can disagree, plus that store's own cut points. An expandable section lists
what every registered regressor predicted for the same row:

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
column, so there is nothing for a shuffled split to leak through and `train_test_split` is
defensible.

What that costs, stated rather than hidden: this answers *what sales should we expect at
this store on this kind of day*, **not** *forecast the next six weeks*. Two adjacent days at
one store are near-duplicates, so a shuffled split still flatters the score a little — much
smaller than a lag leak, not zero.

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

## Results

`compare` does one `train_test_split`, then fits every registered model on the same training
rows and scores it on the same test rows. The only thing that differs between two rows is
the estimator.

```
$ python -m stockout compare --task regression
synthetic: 46,723 open store-days | 50 stores | 2013-01-01 to 2015-12-31

| model             | r2     | mse            | fit_seconds | taught |
|-------------------|--------|----------------|-------------|--------|
| linear            | 0.3029 | 3,953,362.8290 | 0.0590      | yes    |
| ridge             | 0.3029 | 3,953,362.4823 | 0.0220      | yes    |
| lasso             | 0.2717 | 4,130,459.0389 | 0.1220      | yes    |
| polynomial        | 0.6527 | 1,969,903.7190 | 0.3770      | yes    |
| random_forest     | 0.9249 | 426,189.8928   | 13.9660     | no     |
| gradient_boosting | 0.8674 | 751,974.1327   | 3.8860      | no     |
| voting            | 0.8235 | 1,000,895.8538 | 20.1740     | no     |
```

Retail demand is where the extra complexity pays. A straight line manages r² 0.30 because
sales depend on interactions — a promotion on a Saturday is not a promotion plus a Saturday
— and a linear model has no way to represent one. The quadratic expansion doubles that to
0.65 and the forest more than triples it to 0.92.

```
$ python -m stockout compare --task classification
floor: 34.2% (always guess the commonest class)

| model         | accuracy | fit_seconds | taught |
|---------------|----------|-------------|--------|
| logistic      | 0.5987   | 0.1430      | yes    |
| knn           | 0.6790   | 0.0380      | yes    |
| decision_tree | 0.6898   | 0.2510      | yes    |
| random_forest | 0.7551   | 4.3020      | yes    |
| bagging       | 0.7403   | 1.2980      | yes    |
| svc           | 0.7671   | 50.4110     | yes    |
```

The floor is 34.2% because per-store terciles make the three classes near-equal by
construction. Every model clears it, so every one has learnt something; `svc` leads at
76.7% but takes 50 seconds to fit against `random_forest`'s 4 for less than a point.

## Layout

```
src/stockout/            the library — pure, no I/O beyond reading its own data
  cli.py commands.py     five subcommands
  config.py errors.py
  data/
    schemas.py           the canonical column names
    adapters.py          synthetic | rossmann -> one shape
    download.py          the Kaggle fetch
    synth.py synth_stores.py loaders.py
  datasets.py            load, build features, hand back a model-ready frame
  features.py            calendar + store features + the LabelEncoder vocabulary
  targets.py             per-store terciles, fitted on training rows
  models.py registry.py  the thirteen estimators
  evaluate.py            r2 / mse / accuracy, the compare loop
  train.py               fits BOTH models, saves one artefact
  predict.py             one store-day -> sales + demand class
  persistence.py         artefact v3
  plots.py

src/stockout_web/        the app — nothing but I/O
  main.py service.py     startup, and the one deployed ModelService
  auth.py db.py users.py bcrypt, SQLite, roles
  routers/               auth | forecast | admin
  templates/ static/

tests/test_auth.py       10 tests, auth and access control only
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
