# The questions, written before the analysis

Committed before the notebook was opened, so the findings cannot be retrofitted to
whatever the charts happened to show. Answers live in [results.md](results.md).

Each question names what would count as a **no**. A hypothesis that cannot lose is not a
hypothesis, it is a plan to describe whatever happens.

> **Q5 was replaced on 2026-08-18 and does not have that property.** The original asked
> whether stocking to the newsvendor quantile beat stocking to the mean, and the inventory
> simulator that would have answered it was deleted by
> [ADR 0014](decisions/0014-a-scikit-learn-comparison-not-an-inventory-system.md). Its
> replacement was written by somebody who had already seen the comparison table, which is
> exactly the thing the rest of this file exists to prevent. Q1 to Q4 are unchanged and
> predate every chart. Read Q5's verdict with that discount applied.

---

## Q1 — Does forecast accuracy decay with horizon, and how fast?

Backtest the same model at horizons of 7, 14, 28 and 42 days and plot WMAPE against
horizon.

- **Expected:** monotonic decay, steepest between 7 and 14 days.
- **Counts as a no:** WMAPE flat across horizons, which would mean the model is
  predicting a store-level average and ignoring recent history entirely.
- **Why it matters:** the horizon is a business input — it is set by supplier lead time,
  not chosen by the modeller. This curve says what accuracy that lead time buys.

## Q2 — Does seasonal-naive beat a gradient-boosted model on low-volume stores?

Split stores into volume quartiles and compare MASE within each.

- **Expected:** GBM wins overall but loses on the bottom quartile, where each store has
  too few high-signal observations.
- **Counts as a no:** GBM wins uniformly across quartiles.
- **Why it matters:** if it holds, the right answer is a per-store model choice, not one
  model. That is a finding, and it is the kind nobody puts in a tutorial.

## Q3 — How much of the total error comes from a small number of days?

Rank test-set rows by absolute error and plot the cumulative share.

- **Expected:** heavily concentrated — a small share of days carries most of the error,
  clustered on holidays and promotion boundaries.
- **Counts as a no:** error spread evenly across days, so there is no special-case worth
  building.
- **Why it matters:** it decides whether effort belongs in better features or in a
  separate event model.

## Q4 — Does the promotion lift persist after the promotion ends, or reverse?

Compare sales in the seven days after a promotion window against a matched non-promotion
baseline for the same store and weekday.

- **Expected:** a dip. Promotions pull demand forward rather than creating it.
- **Counts as a no:** post-promotion sales at or above baseline, meaning promotions
  genuinely grow demand.
- **Why it matters:** if demand is pulled forward, stocking to a promotion forecast
  overstocks the following week — a forecasting error that only shows up in the
  inventory simulation, never in WMAPE.

## Q5 — Does the classifier add anything over binning the regressor?

Fit the same estimator twice on the same rows: once to predict `sales`, once to predict
the Low/Medium/High class. Bin the regression's output against the same per-store cut
points the classifier was trained on, and score both against the truth.

- **Expected:** the dedicated classifier wins, because it optimises the boundary it is
  scored on rather than a squared error that treats every currency unit alike.
- **Counts as a no:** binning the regression scores at or above the classifier on macro-F1,
  which would mean half the registry answers a question the other half already answered.
- **Why it matters:** it decides whether the classification half of this package is a
  product or a demonstration. `predict._label` already leans on the answer — when the
  classifier abstains on a closed day, the served label falls back to binning the number.
  If binning is as good everywhere, that fallback is the whole product.

*(Replacing the original Q5, which needed the deleted inventory simulator. See the note at
the top of this file: unlike Q1–Q4, this one was written after the comparison table
existed.)*

---

## Rules for answering these

1. Answer them in [`notebooks/01_explore.ipynb`](../notebooks/01_explore.ipynb), write the
   conclusions in [results.md](results.md), and quote the number.
2. **Do not add a sixth question after seeing the data.** If something interesting turns
   up, it goes in results.md under "what we did not ask", clearly marked as unplanned.
3. Report the ones that lose. A results file where all five hypotheses won is evidence
   that the questions were written after the charts.
