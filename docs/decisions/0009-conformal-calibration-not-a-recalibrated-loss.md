# 0009 — Calibrate the quantiles conformally, and leave the raw model alone

> **Superseded by [ADR 0014](0014-a-scikit-learn-comparison-not-an-inventory-system.md).**
> The code this records — the quantile models, the conformal calibration and the
> inventory simulator — was removed in `c7fadcb`. The record is kept because a
> decision log that is edited to match the present is not a log; nothing below
> describes code that ships today.

## Situation

ADR 0007 made the service level falsifiable: a 0.9 quantile should be exceeded 10% of the
time, and `metrics.coverage` measures whether it is. The measurement came back and the
answer was no. On the newest fold of the committed sample every level under-covers, and
the top of the grid is not the worst of it:

| quantile | empirical | gap |
|---|---|---|
| 0.50 | 0.357 | −0.143 |
| 0.75 | 0.564 | −0.186 |
| 0.80 | 0.607 | −0.193 |
| 0.90 | 0.721 | −0.179 |
| 0.95 | 0.843 | −0.107 |
| 0.99 | 0.893 | −0.097 |

A 0.75 that delivers 0.564 is not a service level, it is a number that happens to sit
between 0.5 and 1. Everything downstream inherits the error: the frontier prices a policy
nobody selected, and the cheapest row lands at 0.90 rather than at the 0.75 the cost pair
derives — not because the business wants 0.90, but because the model's 0.90 is really
about a 0.72.

The pinball fit is not doing anything wrong. It is asked to bracket the spread of demand
six weeks out, and the only spread it ever saw during training was the spread of a
training window. Trees do not extrapolate variance.

Three ways out were available.

**Retune the boosters** — deeper trees, less regularisation, more rounds. Fixes coverage
by accident if at all, and every hyperparameter that moves is one more thing to defend.

**Fit a distribution to the residuals** and take its quantiles. That re-imports the normal
assumption ADR 0007 exists to avoid, one layer further down where it is harder to see.

**Conformal calibration** — measure the residuals on data the model has not seen and shift
each quantile by what they say.

## Decision

`models/conformal.py::ConformalQuantileForecaster` wraps a quantile forecaster and adds
one offset per level, computed as a conformal order statistic of the residuals on a
held-out tail of the training window. It fits no distribution: no normality, no variance
model, nothing taken from the residuals but an order statistic of them.

**It does not carry the split-conformal guarantee, and must not be described as though it
does.** That theorem covers the model whose residuals were measured. The model deployed
here is refitted on more data — see *two models are fitted* below — so the residuals and
the predictions come from different estimators, exchangeability fails, and the marginal
coverage statement stops being proven. What remains is a heuristic calibration with a
directional argument behind it and an empirical result in front of it. Every coverage
figure in this repository is measured on held-out data and printed by
`stockout calibration`; none of them is a guarantee, and the one table below where the
method makes things worse is the reason that distinction is worth keeping.

Four choices inside it are not obvious and are therefore recorded.

**The calibration window is one horizon long, and it is the tail.** Residuals gathered one
week ahead say nothing about the spread six weeks ahead, which is the entire defect. The
window is entered from the same distance the test window will be, and it sits immediately
before it, so it is the most recent evidence available.

**The conformity score is scaled, not raw.** A pooled offset in currency units
over-corrects a quiet store and under-corrects a busy one. Each residual is divided by the
model's median prediction for that row before pooling, so the offset is a relative
correction and travels across stores. This is the same heteroscedasticity ADR 0007 cites
as the reason not to use `z * sigma`, and it would be strange to invoke it there and
ignore it here.

**Two models are fitted, not one — and this is what costs the guarantee.** Split conformal
wants the offsets measured on the model that gets deployed; the deployed model wants every
day of history, and its lag features are built by *position*, so a hole punched in its
calendar would misalign them silently. Rather than trade one for the other, a probe model
is fitted on the inner window and scored on the calibration tail, and the deployed model is
fitted on all of it.

The alternative — serving predictions from the probe — would keep the theorem intact and
reintroduce the misalignment the two-fit design exists to avoid: the probe's retained
history stops 42 days before the test window, and a positional lag reaching across that
hole lands on the wrong date. A silently wrong feature is worse than an unproven bound, so
the bound is the thing given up, and it is given up in writing rather than by omission.

The direction of the resulting error is arguable but not provable. The probe trains on
less data, so its residuals should be no smaller than the deployed model's and the offsets
it yields should if anything over-correct — which is the safe direction for a service
level. "Should" is doing real work in that sentence, and the measured coverage below is
what the decision actually rests on.

**The raw model stays exactly as it was.** `GbmQuantileForecaster` is a faithful pinball
fit and its under-coverage is a true fact about pinball fits at long horizons.
`gbm_conformal` is a separate name in the registry, `stockout calibration` prints both
tables side by side, and the reader gets to see the size of the correction rather than a
model that quietly already had it applied.

## Consequence, and it is the one that matters

Calibration moves the cost-minimising service level from **0.90 to 0.80**, and cuts the
total cost of the newest fold from 44,271 to 43,456. ADR 0007 predicted that the optimum
should sit at the critical ratio the cost pair derives, which is 0.75. It did not, and the
reason turns out to have been the under-coverage rather than the theory. Fixing the
coverage moved the answer towards the prediction. That is the closest thing to a
confirmed hypothesis this repository currently holds.

Pinball loss improves at every level too — 350.8 to 337.2 at the median, 74.4 to 42.3 at
the 0.99 — so this is not coverage bought by making the forecast worse.

## Cost

**Fitting takes twice as long**, because the probe is a full set of boosters that is then
thrown away. On the committed sample that is seconds; on Rossmann it is not nothing.

**The correction is marginal, not conditional.** Coverage is corrected on average across
the pooled rows. A particular store, or December, can still be badly covered, and this
layer will not notice. Conditional coverage needs either per-group calibration — which
runs straight into the row counts below — or a Mondrian scheme, and neither is here.

**The correction needs rows, and below a floor it makes things worse.** Measured across
four synthetic draws at a 42-day calibration window:

| stores | calibration rows | raw worst gap | calibrated worst gap | raw pinball | calibrated pinball |
|---|---|---|---|---|---|
| 2 | 70 | 0.107 | **0.171** | 188.9 | **195.6** |
| 3 | 105 | 0.195 | 0.148 | 119.1 | 101.1 |
| 4 | 140 | 0.193 | 0.121 | 225.8 | 200.1 |
| 5 | 175 | 0.259 | 0.236 | 198.4 | 176.3 |

On two stores the calibration is worse than no calibration, on both measures. Seventy
pooled residuals are not enough to estimate six quantiles, and the offsets are fitting
noise. No threshold is enforced, because any number chosen from four draws would be a
magic constant wearing a justification; what is enforced is that `calibration_rows` is
public and the command line prints it.

**Levels near 1 can saturate, and are named when they do.** The finite-sample correction
lands strictly inside the sample only when `ceil((n + 1) * q) < n`, which for a 0.99 first
holds at 199 rows. Below that the offset for that level is the largest residual in the
window — the worst day that happened, not an estimate of a quantile. `saturated_quantiles`
lists them and `stockout calibration` prints the warning, because a service level resting
on one observation should say so out loud.

**It is not enough.** After calibration the 0.90 still covers 0.779. The direction is
right and the magnitude is not; the remaining gap is the distribution shift between a
42-day calibration window and the 42-day test window that follows it, and no amount of
conformal arithmetic on the first will invent knowledge of the second.
