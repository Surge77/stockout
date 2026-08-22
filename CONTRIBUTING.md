# Contributing

## What this project is

**stockout — Retail Sales and Product Demand Prediction.** It predicts store-day sales
and classifies demand as Low, Medium or High, on retail store-day data.

The domain is fixed. A pull request that points this package at non-retail data will be
closed regardless of how clean it is: a constraint on technique is never a licence to
change what the project predicts.

## The one rule about technique

**Only concepts taught in `ML/Resource Files` and `ML/Personal Files` may be used, plus
the three documented exceptions.**

That is the whole point of this repository as of `0.7.0`. The allowed list lives in
`scripts/check_taught.py` and is mirrored in the README. Before opening a pull request:

```bash
python scripts/check_taught.py
```

It must exit 0. `RandomForestRegressor`, `GradientBoostingRegressor` and `VotingRegressor`
are the three exceptions, added in `0.8.0` by request; they carry `taught=False` in the
registry and every table says so. `--strict` rejects those three as well and is the way to
see the current exception list.

A change that adds a **fourth** untaught estimator — `GridSearchCV`, `ColumnTransformer`,
`cross_val_score`, `AdaBoost`, another regressor twin — needs the same treatment: an entry
in the README's exceptions table, `taught=False` on its `ModelSpec`, and an entry in
`EXCEPTIONS` in `scripts/check_taught.py`. Adding one silently is the thing `0.7.0` existed
to undo.

## Setup

```bash
pip install -e ".[dev,notebook]"
```

## Before you push

```bash
ruff check .                     # lint only — never `ruff format`
pyright                          # must be 0 errors
python scripts/check_taught.py   # no undocumented sklearn import

python -m stockout compare --dataset salary --task regression
python -m stockout compare --dataset loans --task classification
python -m stockout compare --dataset loans --task regression
python -m nbconvert --execute --to notebook --inplace notebooks/01_explore.ipynb
```

There is no test suite. The commands above are the verification, and the notebook must
execute top to bottom with no error output.

## Style

- Files stay under 300 lines. Split by responsibility before adding to a long one.
- Type hints on every public signature.
- `X_train` and `Y_test` keep their capitals — that is scikit-learn's convention and the
  notebooks' own naming, and `ruff` is configured to allow it.
- Comment *why*, not *what*. A comment explaining a hidden constraint earns its place; one
  restating the line below it does not.

## Commits

Conventional commits: `<type>(<scope>): <description>`, where type is one of `feat`, `fix`,
`docs`, `refactor`, `chore`, `perf`, `test`.

```
feat(models): register the six taught classifiers
fix(persistence): refuse an artefact written by an older version
docs(readme): state which concepts the package is allowed to use
```

Branch off `main` as `feature/<description>` or `fix/<description>`. Never commit to `main`
directly.
