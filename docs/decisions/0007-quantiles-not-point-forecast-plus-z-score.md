# 0007 — Forecast the quantile directly, not a mean plus a safety stock

## Situation

The textbook safety-stock formula is `S = mu * L + z * sigma * sqrt(L)`: forecast the
mean, estimate the error standard deviation, multiply by a `z` from the normal table for
the service level you want.

It assumes forecast errors are normal, symmetric, and of constant variance. Retail demand
errors are none of those. Variance scales with level (a store selling 20,000 a day has
larger absolute errors than one selling 4,000), and the right tail is longer than the left
because demand can spike far above the mean but cannot fall below zero.

## Decision

Fit LightGBM with `objective="quantile"` at several `alpha` values and read the stocking
level straight off the predicted quantile. No distributional assumption is made, and
whether the answer is right becomes **measurable**: a well-calibrated 0.9 quantile is
exceeded 10% of the time, and `metrics.coverage` checks it.

The quantile to fit is not a hyperparameter. It is the **newsvendor critical ratio**
`Cu / (Cu + Co)` — implemented in `inventory/policy.py::critical_ratio` — where `Cu` is
the cost of being one unit short and `Co` the cost of carrying one unit too many. With the
default cost pair (a lost sale hurting three times as much as carried stock) that is 0.75,
and the model should be fitting the 0.75 quantile. Change the costs and the target moves
with them, which is the point: the service level is derived from the business rather than
tuned by a grid search.

Question Q5 in [questions.md](../questions.md) tests whether this actually beats the
normal approximation. A **no** is a real possible outcome and would be the most
interesting result the project could produce.

## Cost

**Five models instead of one.** Training and storage multiply by the number of quantiles,
and each is fitted independently.

**Quantile crossing.** Independent fits can put the 0.8 prediction above the 0.9 one,
especially in the tails. The fix is to sort each row's quantile vector — and to report how
often crossing happened, rather than sorting silently and pretending it did not.

**Harder to explain than `mu + z * sigma`**, which every planner already knows. That is
worth the trade only because the calibration check makes the claim falsifiable, and the
`z`-based one is not.
