# Results

> **The five questions are still unanswered, and that is a data problem rather than a
> code one.** Every path they need now exists and runs; what does not exist is the
> Rossmann file, which needs a Kaggle account and a signed acceptance of the competition
> rules. The only data a fresh clone has is synthetic, and this repository's own rule is
> that the synthetic sample *"is used to test machinery, never to support a finding"*.
> Answering Q1–Q5 from a generator would be answering a question about
> `stockout.data.synth`, then printing it under a heading about retail.

Every answer needs three things: **a number**, **the chart or table it came from**, and
**a verdict on the hypothesis** — held, lost, or inconclusive. "It seems that" is not a
verdict.

---

## Machinery, demonstrated — not findings

Everything in this section is a statement about the generator. It is here because "the
pipeline runs end to end and produces coherent numbers" is worth evidencing, and because
two of these numbers are bad in an interesting way.

Committed synthetic sample, 4 stores, 730 days, 5 rolling-origin folds, 42-day horizon.

| Model | Mean WMAPE | MASE | RMSPE |
|---|---|---|---|
| `gbm` | 0.0793 | **0.533** | 0.1018 |
| `gbm_quantile` | 0.0811 | 0.545 | 0.1053 |
| `seasonal_naive` | 0.1489 | 1.000 | 0.1931 |
| `moving_average` | 0.1606 | 1.080 | 0.2195 |
| `naive_last` | 0.1695 | 1.144 | 0.2177 |

The gradient-boosted model beats the baseline by 46.7%, and the number should be
discounted heavily. This generator's promotion calendar alternates on a fixed weekly
cycle and its weekday multipliers are constants, so a model with calendar features is
handed structure that real demand only approximates. `seasonal_naive` is also penalised by
exactly that fixed alternation, because same-weekday-last-week always lands on the
opposite promotion state.

### The decision layer

`python -m stockout frontier`, store 1, newest fold, costs `Cu = 3` short and `Co = 1`
carried:

| quantile | fill rate | cycle service | stockout days | holding | shortage | total |
|---|---|---|---|---|---|---|
| 0.50 | 0.945 | 0.000 | 22 | 8,794 | 56,291 | 65,086 |
| 0.75 | 0.963 | 0.000 | 18 | 15,238 | 37,626 | 52,864 |
| 0.80 | 0.971 | 0.000 | 18 | 17,720 | 30,274 | 47,994 |
| 0.90 | 0.980 | 0.167 | 10 | 23,907 | 20,364 | **44,271** |
| 0.95 | 0.984 | 0.167 | 8 | 30,102 | 16,318 | 46,420 |
| 0.99 | 0.990 | 0.167 | 6 | 38,926 | 10,544 | 49,470 |

Total cost is U-shaped in the service level, which is the shape the whole argument
depends on: accuracy is monotone, cost is not, so "as accurate as possible" and "as cheap
as possible" are different instructions.

---

## What didn't work

*Two things, both kept because a dead end costs a day whether or not it is written down.*

### The quantile models are not calibrated out of sample

Measured on the newest fold, trading days only:

| Nominal quantile | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| Actually covered | 0.357 | 0.564 | 0.607 | 0.721 | 0.843 | 0.893 |

Every level under-covers, and the gap widens toward the middle of the distribution. In
sample the same 0.9 model covers 0.865, so this is generalisation error rather than a
fitting bug: quantile regression fits the spread of the residuals it was shown, and the
spread at a 42-day horizon on held-out data is wider than that.

It has a direct consequence, and it is the most interesting number in the repository.
**The cost-minimising service level is 0.90, but the newsvendor critical ratio derived
from the cost pair is 0.75.** ADR 0007 predicted that fitting the quantile directly would
make the claim falsifiable, and on this data it is falsified: the nominal quantile is not
the achieved service level, so the theoretical optimum lands in the wrong place. You have
to over-ask by roughly one grid step to get what you asked for.

The tempting fix is to shift the predictions until coverage matches. It is not applied,
because a post-hoc calibration would hide the fact that this happened, and Q5 is precisely
the question of whether the derived quantile beats the alternatives — a question that
cannot be asked honestly of a model that has been nudged toward the answer.

Quantile crossing affected **42.9%** of out-of-sample rows and is sorted before use. That
is high, and consistent with the same underlying problem: six independently fitted
boosters disagreeing about a spread none of them has enough held-out signal to pin down.

### The first efficient frontier had no trade-off in it

`frontier` originally converted each quantile into an `(R, S)` base-stock level covering
the lead time plus a review period, and handed that to a simulator that refills daily. On
store 1 that is a level of 96,159 against a mean daily demand of 8,132 — **11.8 days of
cover, held every single day**. Every service level returned a fill rate of 1.0000 with
zero stockouts, total cost became monotone in the quantile, and the chart said nothing.

The error was conceptual, not arithmetic: both halves were individually correct and could
not be composed. Fixed by naming what the simulator actually models — a repeated
single-period newsvendor, which is what the critical ratio solves — and stocking to the
day's own quantile. Written up in
[ADR 0008](decisions/0008-the-simulator-has-no-shipping-lag.md), with
`test_the_frontier_stocks_to_the_forecast_itself_not_to_a_multi_day_cover` to stop it
recurring quietly.

---

## Q1 — Accuracy decay with horizon

*Unanswered.* The backtest accepts `--horizon`, so this is one loop over 7, 14, 28 and 42
once there is real data to loop over.

**Number:**
**Evidence:**
**Verdict:**

## Q2 — Seasonal-naive vs GBM on low-volume stores

*Unanswered.* Both models exist and both are reachable from `--model`; what is missing is
a store population with genuinely varied volume. The generator's four stores are drawn
from one distribution on purpose.

**Number:**
**Evidence:**
**Verdict:**

## Q3 — Concentration of error across days

*Unanswered.*

**Number:**
**Evidence:**
**Verdict:**

## Q4 — Post-promotion demand

*Unanswered.* The generator has no post-promotion dip to find — its promotion lift is a
constant multiplier with no payback period — so a null result here would measure the
generator and nothing else.

**Number:**
**Evidence:**
**Verdict:**

## Q5 — Newsvendor quantile vs mean-plus-safety-stock

*Unanswered*, and the calibration result above is the reason to expect this one to be
close. The machinery to answer it is complete: `frontier` prices any set of quantile
columns, so the comparison is one more column built from `gbm` plus `z * sigma`.

**Number:**
**Evidence:**
**Verdict:**

## What we did not ask

Anything interesting found while answering the five. Kept separate so that planned
findings and lucky ones are never confused.
