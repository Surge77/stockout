# TODO

Open work on `feature/product-demand`, as of 2026-08-26.

State at the last commit: 139 tests green, `pyright` and `ruff` clean, working tree clean.
Run the app with `uvicorn stockout_web.main:app` on port 8000.

**The deployed artefact decides which pages exist.** A product artefact offers `/products`
and hides `/what-if`; a store-day artefact does the reverse. Switch from the **dataset
dropdown on `/admin`** (36s for either synthetic source) or with
`python -m stockout train --dataset <name>`. Rossmann is over an hour, so it is a CLI job.

| dataset | r² | accuracy | floor |
|---|---|---|---|
| `rossmann` | 0.8296 | 0.6649 | — |
| `synthetic_products` | 0.9058 | 0.4927 | **0.455** |

---

## 1. The product classifier is barely above chance — medium

`random_forest` scores **0.4927** on `synthetic_products` against a **45.5%** floor. That
is 3.8 points, and `logistic` at 0.3147 is *below* the floor. Only `svc` (0.5664, on a
capped 50,000-row sample) clears it meaningfully.

The floor is not the ~33% per-entity terciles produce by construction. Those cut points are
fitted on the training rows; on a chronological hold-out the residual distribution drifts
and one band grows in the test period. Worth understanding before treating this as a
modelling problem — it may be a *labelling* problem, i.e. that residual terciles fitted on
one period do not describe the next one.

The shelf page states the accuracy and the measured floor on screen, so nothing is
overclaimed today. Two things worth trying:

- Deploy `svc` for classification on product datasets, and report the row cap honestly.
- Re-cut the terciles per period rather than once on the training half, and measure whether
  the drift is the whole story.

## 2. What-if page: it is misleading on real Rossmann data — medium

Unchanged from before, and untouched by the product work. Measured leave-one-shop-out on
Rossmann, five held-out shops:

| configuration | mean r² | per shop |
|---|---|---|
| what the page **reports** (raw-sales model) | −3.084 | −7.43, 0.61, −9.24, 0.68, −0.03 |
| what the page **deploys** (residual + chain line) | −1.194 | −4.55, 0.67, −3.28, 0.78, 0.41 |

Three defects:

- **The reported score measures the wrong model.** `train._unseen_store_r2` fits on raw
  sales while the deployed cold model is fitted on residuals with the chain-wide line added
  back. Fix the measurement first — everything else depends on it.
- **The shop-group input cannot be answered.** Rossmann never published what the groups mean.
- **A point estimate implies precision that is not there.** Show a range.

Note the page is now *correctly absent* on product deployments, which is a different thing
from being fixed on store-day ones.

## 3. 351 shops with a competitor and no opening date — medium

`store.csv` gives a `CompetitionDistance` for 1,112 of 1,115 shops but no
`CompetitionOpenSince*` for 354 of them. `store_features._age_in_months` fills those with
`0`, which the model reads as *a competitor opened this very month* — the freshest possible
competition — for 32% of rows, on a feature carrying roughly a sixth of model importance.

One shop also claims its competitor opened in 1900, giving an age of 1,386 months.

Standard fix is a missing-indicator column beside the age. That changes the feature matrix,
so it invalidates the artefact and every published number — measure before and after.

## 4. README store-day tables are stale — slow, not hard

The `synthetic` tables in `## Results` predate the switch to real data. The product tables
and the Rossmann prose figures are current; these are not. Regenerate:

```bash
python -m stockout compare --dataset rossmann --task regression
python -m stockout compare --dataset rossmann --task classification
python -m stockout leakage --dataset rossmann
```

Each takes several minutes.

## 5. The Rossmann artefact is 2.6 GB — low

`random_forest` with sklearn's default 100 estimators over 670,422 rows dominates it. It
loads in seconds and works, but it is an awkward thing to keep on disk, and `artifacts/` is
gitignored so it is unrecoverable if overwritten. Capping `max_depth` or `n_estimators`
would shrink it; measure what that costs in r² first.

## 6. Retraining a large dataset from the dashboard will outlast the browser — medium

The dataset dropdown makes every source deployable from `/admin`, and both synthetic
sources fit in about 36 seconds. Rossmann takes over an hour: `train._unseen_store_r2`
refits the regressor once per held-out shop over 670,422 rows. The request is off the event
loop, so the app keeps answering everyone else, but the admin's own tab will time out long
before the fit lands — and the retrain still completes and deploys, so the result is a
confusing success that looks like a failure.

The form marks those datasets **slow** and points at the CLI, which is honest but not a
fix. A real one is a background job with a status the page can poll. Worth doing only if
deploying the competition data from the web is actually wanted.

---

## Done in this branch

- **Product demand.** A row can be one shop-item-day with `sales` in units. Two new sources
  (`synthetic_products` committed, `store_item_demand` fetched), `store_code`/`item_code` as
  separate features, `ranking.py` for the whole shelf in one pass, `/products`, and
  `predict --item` / `products` on the CLI.
- **`entity_code` did not survive serving.** It fell back to `pd.factorize` for any
  non-numeric id, which numbers whatever frame it is handed — so every store-item request
  answered about whichever series sorted first. The vocabulary now travels in the artefact.
  Rossmann keeps its numeric fast path, so its codes and scores did not move.
- **The comparison list answered a different question from the headline.** It returned
  residuals while the headline added the trend back, and dropped both holiday fields. Both
  now go through `predict.compare_row`.
- **`ModelService.load` no longer retrains over an artefact it cannot read**, which had
  silently replaced a Rossmann deployment with a synthetic one on a version bump.
- **The dataset is deployable from `/admin`.** It passed `dataset=artifact.dataset` and had
  no field for anything else, so the whole product half was reachable only from a terminal.
- **Startup is 0.4s, not thirteen minutes.** The seven comparison regressors and the admin
  tables are warmed in the background after the app is answering, not before it will.
- **`.bar` was defined twice** — the shelf bars took the site header's class name and broke
  every page's header. `tests/web/test_stylesheet.py` fails on any duplicate class now.
