# TODO

Open work on `feature/strip-to-taught-concepts`, as of 2026-08-24.

State at the last commit: 89 tests green, `pyright` and `ruff` clean, working tree clean.
Deployed artefact is trained on the real Rossmann data (r² 0.8296, accuracy 0.6649).
Run the app with `uvicorn stockout_web.main:app` on port 8000.

---

## 1. What-if page: it is misleading on real data — medium

Measured leave-one-shop-out on Rossmann, five held-out shops:

| configuration | mean r² | per shop |
|---|---|---|
| what the page **reports** (raw-sales model) | −3.084 | −7.43, 0.61, −9.24, 0.68, −0.03 |
| what the page **deploys** (residual + chain line) | −1.194 | −4.55, 0.67, −3.28, 0.78, 0.41 |

Negative means worse than answering with the average. Three of five shops score well; two
fail badly. The page is informative for ordinary shops, wrong for unusual ones, and
presents both with the same confidence.

Three separate defects:

- **The reported score measures the wrong model.** `train._unseen_store_r2` fits on raw
  sales while the deployed cold model is fitted on residuals with the chain-wide line
  added back. Fix the measurement first — everything else depends on it.
- **The shop-group input cannot be answered.** The form asks which of four groups a
  hypothetical shop belongs to, but Rossmann never published what the groups mean.
- **A point estimate implies precision that is not there.** Show a range.

Recommendation: keep the page and make the difficulty the point. An honest negative score
with an explanation of why four attributes cannot determine a shop's trading level is a
better artefact than a fabricated positive one. Deleting it loses the only page whose
answer is not a lookup.

## 2. 351 shops with a competitor and no opening date — medium

`store.csv` gives a `CompetitionDistance` for 1,112 of 1,115 shops but no
`CompetitionOpenSince*` for 354 of them. `features._age_in_months` fills those with `0`,
which the model reads as *a competitor opened this very month* — the freshest possible
competition — for 32% of rows, on a feature carrying roughly a sixth of model importance.

One shop also claims its competitor opened in 1900, giving an age of 1,386 months.

Standard fix is a missing-indicator column beside the age, so a tree can split on *is this
known* rather than treating unknown as freshest. That changes the feature matrix, so it
invalidates the artefact and every published number — measure before and after.

## 3. README tables are stale — slow, not hard

Every table in `## Results` is synthetic and now understates the project. Regenerate:

```bash
python -m stockout compare --dataset rossmann --task regression
python -m stockout compare --dataset rossmann --task classification
python -m stockout leakage --dataset rossmann
```

Each takes several minutes; run them in the background. The classification table has never
been run on real data at all — the deployed classifier scores 0.6649 against a ~37% floor,
and no other classifier has been tried.

Also update the date-policy examples: Rossmann ends **2015-07-31**, five months earlier
than the synthetic sample, so the worked dates in that section are wrong.

## 4. Standing rule for any copy written here

No invented labels for anonymised fields. Assortment meanings (basic / extra / extended)
are documented by the competition and may be used. Shop groups A–D are **not** — Rossmann
never said what they mean. Explain what the field is and why the model uses it; never name
it. Prefer facts derived from the data over descriptions made up to sound helpful.
