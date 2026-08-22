# stockout

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Checked with pyright](https://img.shields.io/badge/types-pyright-blue)](https://github.com/microsoft/pyright)

Ten scikit-learn models on two datasets, each behind the same `Pipeline`, compared in one
table. Every concept in this package appears in the course notebooks under
`ML/Resource Files` — and nothing else does.

> **The name is historical.** Up to `v0.6.0` this was a retail demand forecaster on the
> Rossmann store-sales data, with rolling-origin backtesting, lag features, leakage guards
> and a FastAPI web app. All of that used machinery the course does not teach, so `0.7.0`
> removed it. See [CHANGELOG.md](CHANGELOG.md) for what went and why.

```bash
pip install -e .

python -m stockout describe --dataset salary              # rows, dtypes, nulls, summary
python -m stockout models                                 # what --model accepts
python -m stockout compare --dataset salary               # the four regressors
python -m stockout compare --dataset loans --task classification   # the six classifiers
python -m stockout train --dataset salary --model lasso   # fit and save
python -m stockout predict --interactive                  # prompt, then predict
```

## The concepts this package is allowed to use

Taken from the code cells of the fourteen notebooks in `ML/Resource Files` and
`ML/Personal Files`, plus `Task_Regression.pdf`. This is the complete list.

| Area | Used here |
|---|---|
| split | `train_test_split` |
| pipeline | `Pipeline` |
| preprocessing | `StandardScaler`, `PolynomialFeatures` |
| linear models | `LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression` |
| trees & ensembles | `DecisionTreeClassifier`, `RandomForestClassifier`, `BaggingClassifier` |
| neighbours & kernel | `KNeighborsClassifier`, `SVC` |
| metrics | `r2_score`, `mean_squared_error`, `accuracy_score` |
| libraries | pandas, numpy, matplotlib, seaborn |

`LabelEncoder` and `TfidfVectorizer` are taught and are deliberately **not** used: every
column in both datasets is numeric, so there is no categorical column to encode and no text
column to vectorise. Inventing one to justify the import would be the same
over-engineering this release exists to remove.

Nothing outside that table is imported anywhere in `src/`. To check:

```bash
grep -rEho "from sklearn[a-zA-Z_.]* import [A-Za-z_, ]*" src/ \
  | grep -vE "train_test_split|Pipeline|StandardScaler|PolynomialFeatures|LabelEncoder|LinearRegression|Ridge|Lasso|LogisticRegression|DecisionTreeClassifier|RandomForestClassifier|BaggingClassifier|KNeighborsClassifier|SVC|TfidfVectorizer|r2_score|mean_squared_error|accuracy_score"
```

That prints nothing.

## The data

Both files come from `ML/Resource Files/Datasets` and were converted from `.xlsx` to `.csv`
once, using the standard library, so the package needs no `openpyxl`. Both are committed,
so a fresh clone runs offline.

**`data/salary.csv`** — 10,000 rows, from `Task_R.xlsx`. This is the assignment's dataset:
its ten feature columns are exactly the ten `input()` prompts on the last page of
`Task_Regression.pdf`.

| | |
|---|---|
| features | `age`, `experience_years`, `education_level`, `certifications`, `projects_completed`, `interview_score`, `technical_test_score`, `communication_score`, `internships`, `hours_per_week` |
| regression | `salary` |

**`data/loans.csv`** — 10,000 rows, from `Loan_Regression_Classification.xlsx`. Carries a
paired problem: whether the loan was approved, and how much was granted.

| | |
|---|---|
| features | `Age`, `Monthly_Income`, `Credit_Score`, `Employment_Years`, `Existing_Debt`, `Property_Value` |
| classification | `Loan_Approved` (0 or 1) |
| regression | `Loan_Amount` (zero exactly when `Loan_Approved` is 0) |

## The models

Every one is `Pipeline([("Scaler", StandardScaler()), ("Model", estimator)])` — the shape
the notebooks use throughout. The scaler sits in front of the tree models too; it is a
no-op for them, and one pattern is easier to read than two.

The regression hyperparameters are the assignment's, not invented here.

| Task | Name | Estimator | From |
|---|---|---|---|
| regression | `linear` | `LinearRegression()` | PDF p.1 |
| regression | `ridge` | `Ridge(alpha=1)` | PDF p.4 |
| regression | `lasso` | `Lasso(alpha=100)` | PDF p.3 |
| regression | `polynomial` | `PolynomialFeatures(degree=2)` → scaler → `LinearRegression()` | PDF p.2 |
| classification | `logistic` | `LogisticRegression(max_iter=1000)` | `LR.ipynb` |
| classification | `knn` | `KNeighborsClassifier()` | `7. KNN.ipynb` |
| classification | `decision_tree` | `DecisionTreeClassifier()` | `DT.ipynb` |
| classification | `random_forest` | `RandomForestClassifier()` | `SVM.ipynb` |
| classification | `bagging` | `BaggingClassifier()` | `SVM.ipynb` |
| classification | `svc` | `SVC()` | `SVM.ipynb` |

The registry is in order of complexity, simplest first, and the results table keeps that
order. A table sorted by score answers *which won*; this order also answers *did the extra
complexity pay*.

Only the classifier forms are here, because only the classifier forms are taught.
There is no `DecisionTreeRegressor`, no `SVR`, no `KNeighborsRegressor`.

## Results

`compare` does one `train_test_split`, then fits every registered model on the same
training rows and scores it on the same test rows. The only thing that differs between two
rows of the table is the estimator.

```
$ python -m stockout compare --dataset salary --task regression

salary · regression -> salary · 4 models · 2,000 held-out rows

| model      | r2     | mse             | fit_seconds |
|------------|--------|-----------------|-------------|
| linear     | 0.9805 | 10,988,772.3710 | 0.2450      |
| ridge      | 0.9805 | 10,988,812.1897 | 0.0040      |
| lasso      | 0.9803 | 11,088,696.4602 | 0.0030      |
| polynomial | 0.9941 | 3,347,614.8693  | 0.0280      |

best: polynomial (r2=0.9941)
```

`Task_Regression.pdf` reports 0.9805 for the linear pipeline and 0.9940 for the polynomial
one, so the two models with no `alpha` to differ on reproduce the assignment exactly. Ridge
and Lasso land within 0.0007 of its numbers; the PDF does not state its `random_state`.

```
$ python -m stockout compare --dataset loans --task classification

| model         | accuracy | fit_seconds |
|---------------|----------|-------------|
| logistic      | 0.9985   | 0.0340      |
| knn           | 0.9620   | 0.0080      |
| decision_tree | 0.9430   | 0.0180      |
| random_forest | 0.9770   | 0.4940      |
| bagging       | 0.9655   | 0.1110      |
| svc           | 0.9885   | 0.1310      |
```

The simplest model wins, which is the useful result: approval is close to a linear function
of credit score and income, and the five more complex models spend their extra capacity on
noise. `Loan_Amount` is the harder target — the linear models manage r² ≈ 0.69 against the
polynomial pipeline's 0.85, because the amount is exactly zero whenever the loan was
refused and that kink is not something a straight line can follow.

## Serving one prediction

`train` fits a model on the training split and writes it, with its held-out scores, to
`artifacts/model.joblib`. `predict` reads it back.

```bash
python -m stockout train --dataset salary --model lasso

python -m stockout predict --set age=29 --set experience_years=5 ...
python -m stockout predict --interactive
```

Interactive mode is step 5 of the assignment: it prompts for each feature in turn and
prints the prediction.

```
Enter Your age: 29
Enter Your experience_years: 5
...
Predicted Salary: 76,642.09
```

`joblib.load` executes code in the file it reads, so `persistence.load` refuses any path
outside `artifacts/`. That is a guard against a careless caller, not a sandbox — an
artefact is exactly as trustworthy as whoever wrote it.

## Layout

```
src/stockout/
  cli.py           argparse: describe | models | compare | train | predict
  commands.py      what each subcommand does
  config.py        paths, RANDOM_SEED, TEST_SIZE
  datasets.py      the two csvs, their columns, and each task's target
  models.py        the ten estimators as Pipeline factories
  registry.py      name -> factory, per task
  evaluate.py      r2 / mse / accuracy, the compare loop, the markdown table
  train.py         split -> fit -> score -> save
  predict.py       one row, from flags or from input()
  persistence.py   joblib save/load
  plots.py         matplotlib + seaborn charts for the notebook
notebooks/
  01_explore.ipynb the assignment's 17 steps, then both comparison tables
data/
  salary.csv       10,000 rows, regression
  loans.csv        10,000 rows, classification and regression
```

## Development

```bash
pip install -e ".[dev,notebook]"

ruff check .        # lint
pyright             # types
```

There is no test suite. `0.7.0` removed it along with everything else that was not part of
the taught material; the package is verified by running the commands above and executing
the notebook top to bottom:

```bash
python -m nbconvert --execute --to notebook --inplace notebooks/01_explore.ipynb
```

## Licence

MIT — see [LICENSE](LICENSE).
