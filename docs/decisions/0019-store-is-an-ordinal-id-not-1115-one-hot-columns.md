# 0019 — `store` is an ordinal id, not 1115 one-hot columns

## Situation

`store` is an integer that identifies a shop. It is categorical in every sense that
matters — store 262 is not twice store 131 — and the textbook move for a categorical
column is `OneHotEncoder`. Rossmann has 1115 of them.

The textbook move is wrong here, for four independent reasons, and it is worth listing
them separately because each one alone would be enough.

**Memory.** 1115 columns across the ~844,000 rows that survive feature building is
**3.77 GB** as float32 and 7.53 GB as float64. The design matrix is pickled into every
`GridSearchCV` worker process.

**Sparsity, which the estimators will not take.** The honest encoding of that matrix is
sparse, and `HistGradientBoosting*` rejects sparse input outright, while `StandardScaler`
with `with_mean=True` densifies it into the figure above. `features/preprocess.py` sets
`sparse_threshold=0.0` on the `ColumnTransformer` precisely so the Tf-idf branch cannot
drag the whole matrix sparse; one-hot encoding `store` would make that setting the thing
that causes the memory bomb.

**Distances stop meaning anything.** On a one-hot matrix, the Euclidean distance between
any two *different* stores is the same constant √2, regardless of which two. Every KNN
neighbourhood and every RBF kernel would be computed against that constant, so the models
in `kernels.py` would be reading store identity as pure noise.

**1115 coefficients nobody can defend.** A linear model would fit one free parameter per
shop, and the resulting object cannot be interrogated, cannot be summarised and will
happily memorise a store that appears in the training window for forty days.

## Decision

`store` is passed through as its integer id, in its own `ordinal` branch of the
`ColumnTransformer` — no scaling, no encoding, no imputation.

The store's *level* is already carried, causally, by `sales_roll_mean_7`,
`sales_roll_mean_28` and `sales_roll_mean_91`. Those are learned from that store's own
history rather than from a free parameter, they update as the shop changes, and they exist
for a store the training window never saw. That is a strictly better representation of
"how busy is this shop" than an identity column, and it is the reason the identity column
can be spared.

The other four categorical columns — `store_type`, `assortment`, `state_holiday_type`,
`day_of_week` — *are* one-hot encoded. They have four, three, four and seven levels. The
decision here is about cardinality, not about principle.

Passing the raw integer through means a tree can still split on it, which is the one thing
the ordinal branch buys that dropping the column entirely would not. A split at
`store <= 640` is meaningless as an ordering and perfectly usable as a partition, and a
forest will use it to isolate a group of shops it has found by their id.

## Cost

**The linear models are handed a column whose magnitude is a lie.** `store` reaches
`LinearRegression` unscaled, as an integer from 1 to 1115, and a linear fit will give it a
coefficient as though store number were a quantity. The coefficient is near zero on this
data because store id genuinely does not trend with sales, but nothing enforces that and
nothing detects it if it stops being true.

**A tree can carve out an individual store**, which is the memorisation the one-hot
encoding was accused of. `min_samples_leaf=50` is the only thing standing in the way, and
it is a blunt instrument.

**Target encoding is not attempted.** Replacing the id with that store's mean sales is the
standard high-cardinality answer, would be strictly more informative than the raw id, and
is a leak the moment it is fitted outside a fold. It is left out because the rolling means
already deliver most of what it would, without the failure mode.

**On the committed four-store sample none of this can be observed.** Four stores one-hot
encoded is four columns, and every argument above is about 1115. The decision is made for
data a fresh clone cannot download without a Kaggle account.
