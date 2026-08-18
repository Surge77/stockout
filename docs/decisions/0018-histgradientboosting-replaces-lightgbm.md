# 0018 — HistGradientBoosting replaces LightGBM

## Situation

Every gradient-boosted number this project ever published came from LightGBM, behind an
optional `gbm` extra. `pip install -e .` gave you a package whose headline model would not
run; `pip install -e ".[gbm]"` gave you one that would. The import was lazy, the failure
was a readable message naming the extra, and CI had a whole job asserting that the
message arrived rather than an `ImportError`.

That machinery existed to protect one thing: the size of the default install. It stopped
being worth it once the package's subject changed. ADR 0014 replaced a single boosted
model with a registry of twenty-one, twenty of which are scikit-learn estimators that the
core dependency already provides. Keeping LightGBM meant one model in the table lived
behind a different install path, a lazy import, an error message, a CI job and a
documented extra — and the other twenty did not.

## Decision

`HistGradientBoostingRegressor` and `HistGradientBoostingClassifier` take the place
LightGBM held, under the registry name `hist_gradient_boosting`. The dependency is
removed, the extra is removed, and the lazy-import machinery with it.

The substitution is close. Both bin continuous features into histograms before splitting,
which is the trick that makes either of them fast on 800,000 rows; sklearn's
implementation is explicitly modelled on LightGBM's. Both handle missing values natively,
which matters here because `competition_distance` ships with real gaps and
`competition_open_months` is NaN wherever the opening date is unknown — the alternative is
imputing a distance for a competitor nobody has located.

What is given up is real and small: LightGBM is faster at the same accuracy, has
categorical-feature support that would let `store_type` skip the one-hot branch, and has
`objective="quantile"` — which mattered a great deal to the version of this project that
forecast quantiles and matters not at all to the version that does not.

`early_stopping` is left at sklearn's `"auto"`, which turns it on above 10,000 rows. When
it is on, the estimator carves its own validation split out of the training rows, so the
seed is load-bearing rather than decoration: without it two runs of the same command
produce different scores.

## Cost

**Every gradient-boosted number in the repository's history is from a different library.**
The `gbm` rows in the old README are not comparable to the `hist_gradient_boosting` rows in
the new one, and no run reproduces both — LightGBM is no longer installable from this
project's own metadata. The old numbers are deleted rather than carried forward.

**ADR 0002 argued for gradient boosting over a sequence model and named LightGBM while
doing it.** The argument survives the substitution intact; the name in it is now wrong.
It is amended by this ADR rather than edited.

**The bare install is heavier and the extra-install path is gone.** A user who wanted the
package without a boosting library no longer has that option, because scikit-learn is a
hard dependency and the boosting lives inside it. That is the trade: one install path
that always works, instead of two of which one is a trap.

**Categorical support is left on the table.** `HistGradientBoosting*` accepts a
`categorical_features` argument that would let the four categorical columns bypass
`OneHotEncoder` entirely, which is both faster and usually slightly better for trees. It is
not used, because the whole registry shares one preprocessor and a per-model preprocessor
would be a second thing to keep in step — the same reasoning `features/preprocess.py`
applies to scaling.
