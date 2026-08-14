# The questions, written before the analysis

Committed before the notebook was opened, so the findings cannot be retrofitted to
whatever the charts happened to show. Answers live in [results.md](results.md).

Each question names what would count as a **no**. A hypothesis that cannot lose is not a
hypothesis, it is a plan to describe whatever happens.

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

## Q5 — Does stocking to the newsvendor quantile beat stocking to the mean?

Run the inventory simulation with the point forecast plus a normal safety stock, and with
the direct quantile forecast, at matched holding cost.

- **Expected:** the quantile policy achieves a higher fill rate at the same cost, because
  demand errors are not symmetric.
- **Counts as a no:** the two policies land on the same frontier, meaning the normal
  approximation was good enough and the quantile models were not worth building.
- **Why it matters:** this is the question the repository is named after. A **no** here is
  the most interesting outcome available and must be reported as loudly as a yes.

---

## Rules for answering these

1. Answer them in [`notebooks/01_explore.ipynb`](../notebooks/01_explore.ipynb), write the
   conclusions in [results.md](results.md), and quote the number.
2. **Do not add a sixth question after seeing the data.** If something interesting turns
   up, it goes in results.md under "what we did not ask", clearly marked as unplanned.
3. Report the ones that lose. A results file where all five hypotheses won is evidence
   that the questions were written after the charts.
