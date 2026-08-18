# 0015 — A subsample is reported, never silent

## Situation

Ten of the twenty-one registered models train on every row that survives feature
building — roughly 800,000 on Rossmann. Three cannot.

`SVR` and `SVC` with an RBF kernel build a kernel matrix over pairs of training rows, so
their cost is between quadratic and cubic in the row count. Measured on a 50-column
standardised matrix, fitting one `SVR`:

| rows | fit time |
|---|---|
| 1,000 | 0.04s |
| 2,000 | 0.15s |
| 5,000 | 1.76s |
| 10,000 | 4.38s |
| 20,000 | 48.07s |

Twenty times the rows costs roughly twelve hundred times the time. Extrapolating to
800,000 is not a wait, it is a refusal. `KNeighbors*` is the opposite shape — it trains
instantly and pays at predict time — so it tolerates about ten times as many rows before
it hurts.

The tempting fix is to subsample inside the model and say nothing. That produces a
comparison table where `svr` scored 0.833 and `ridge` scored 0.842 and the reader has no
way to know the first saw a hundredth of the rows the second did.

## Decision

The cap is a **declared property of the model**, carried on `ModelSpec.sample_rows`, and
it travels with the model into every table that quotes a score.

- `SVR` and `SVC` are capped at `config.SUBSAMPLE_ROWS` (5,000).
- `KNeighbors*` and `LinearSVC` are capped at 50,000.
- Everything else is uncapped, and `sample_rows` is `None`.

`evaluate/comparison.py` reports `train_rows` and `seconds` as columns of the results
table, next to the score, for every model — not only the capped ones. Two row counts, not
one: `train_rows` and `test_rows` are separate columns because they used to collide under
a single `rows` key and the scoring half won, which erased exactly the fact this ADR
exists to preserve.

`models/tuning.py` searches over `training_frame`, which is the capped set. A grid search
run on a different row set than the final fit chooses hyperparameters for a model nobody
builds.

`LinearSVC` is in the registry for this reason and no other. It solves the same problem as
`SVC(kernel="linear")` by a route that is linear rather than quadratic in the rows, so it
sees ten times as many of them — and whether the RBF kernel earns its subsample becomes a
question the table answers rather than one the reader takes on trust.

## Cost

**The capped models are being judged unfairly and the table says so rather than fixing
it.** A fair comparison would cap every model at 5,000 rows, and that would throw away the
main finding available here — that on this problem the models which can use all the data
win because they can use all the data.

**On the committed four-store sample the cap never binds.** 2,395 training rows is below
every cap, so `train_rows` is identical down the column and the mechanism is invisible in
the numbers a fresh clone can produce. It is exercised by `tests/test_registry.py` rather
than by the sample, which is a weaker demonstration than a reader deserves.

**5,000 is a judgement, not a derivation.** It is the largest round number whose fit and
whose grid search both finish in seconds rather than minutes. A different machine would
justify a different number, and no experiment here identifies the point at which more rows
stop buying accuracy.
