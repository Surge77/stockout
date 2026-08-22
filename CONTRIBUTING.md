# Contributing

## The one rule

**Only concepts taught in `ML/Resource Files` and `ML/Personal Files` may be used.**

That is the whole point of this repository as of `0.7.0`. The allowed list is in the
README and is complete. Before opening a pull request, run the guard:

```bash
grep -rEho "from sklearn[a-zA-Z_.]* import [A-Za-z_, ]*" src/ \
  | grep -vE "train_test_split|Pipeline|StandardScaler|PolynomialFeatures|LabelEncoder|LinearRegression|Ridge|Lasso|LogisticRegression|DecisionTreeClassifier|RandomForestClassifier|BaggingClassifier|KNeighborsClassifier|SVC|TfidfVectorizer|r2_score|mean_squared_error|accuracy_score"
```

It must print nothing. A change that adds `GridSearchCV`, `ColumnTransformer`,
`cross_val_score` or a regressor twin will be closed regardless of how good it is — the
previous six releases were exactly that, and removing them is what `0.7.0` was.

## Setup

```bash
pip install -e ".[dev,notebook]"
```

## Before you push

```bash
ruff check .        # lint only — never `ruff format`
pyright             # must be 0 errors

python -m stockout compare --dataset salary --task regression
python -m stockout compare --dataset loans --task classification
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
