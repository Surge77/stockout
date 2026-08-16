# Results

> **The five questions are still unanswered, and that is a data problem rather than a
> code one.** Every cell that answers them exists and has been executed end to end; what
> does not exist is the Rossmann file, which needs a Kaggle account and a signed
> acceptance of the competition rules. The only data a fresh clone has is synthetic, and
> this repository's own rule is that the synthetic sample *"is used to test machinery,
> never to support a finding"*. Answering Q1–Q5 from a generator would be answering a
> question about `stockout.data.synth`, then printing it under a heading about retail.
>
> Running them anyway was not wasted: it found a Q4 design that divides by zero on any
> dense promotion calendar, and it produced a Q5 ranking that would be a **no** if it were
> allowed to count. Both are recorded below under machinery.

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

### Calibration moves the answer, and moves it towards the theory

`python -m stockout frontier --model gbm_conformal`, same store, same fold, same costs:

| quantile | fill rate | stockout days | holding | shortage | total |
|---|---|---|---|---|---|
| 0.50 | 0.953 | 20 | 10,732 | 48,019 | 58,751 |
| 0.75 | 0.976 | 16 | 20,645 | 24,607 | 45,252 |
| 0.80 | 0.980 | 13 | 22,632 | 20,824 | **43,456** |
| 0.90 | 0.985 | 9 | 28,980 | 15,675 | 44,655 |
| 0.95 | 0.989 | 7 | 37,564 | 10,937 | 48,501 |
| 0.99 | 0.996 | 4 | 52,438 | 4,621 | 57,059 |

The minimum moves from **0.90 to 0.80** and gets cheaper, 44,271 to 43,456. That is the
single most load-bearing number produced so far, because it is a *prediction that came
true*: ADR 0007 said the optimum should sit at the critical ratio the cost pair derives,
which is 0.75; the raw model put it at 0.90; the reason turned out to be under-coverage
rather than a wrong theory, and correcting the coverage moved the optimum towards 0.75.
It has not arrived there — 0.75 still under-covers by 0.10 after calibration — and the
remaining distance is the remaining miscalibration. See
[ADR 0009](decisions/0009-conformal-calibration-not-a-recalibrated-loss.md).

### Once delivery takes a week, the critical ratio stops being the target

`python -m stockout frontier --model gbm_conformal --lead-time 7`:

| quantile | fill rate | stockout days | mean on hand | mean on order | total |
|---|---|---|---|---|---|
| 0.50 | 0.970 | 14 | 5,735 | 44,819 | **271,616** |
| 0.75 | 0.995 | 2 | 7,357 | 45,850 | 313,978 |
| 0.80 | 0.997 | 1 | 7,780 | 45,908 | 329,361 |
| 0.90 | 1.000 | 0 | 9,010 | 45,973 | 378,419 |
| 0.95 | 1.000 | 0 | 10,748 | 45,970 | 451,402 |
| 0.99 | 1.000 | 0 | 13,875 | 45,556 | 582,765 |

Costs rise sixfold and the cheapest level collapses to the bottom of the grid. Neither is
a bug. Holding is charged per unit **per day**, and an eight-day protection interval means
carrying roughly eight days of cover instead of one, so the holding term grows by about
that factor while shortage — charged once per lost sale — does not. `Cu / (Cu + Co)` is
the optimum for a *single-period* decision and stops being the optimum the moment the
protection interval is longer than a day. `policy.critical_ratio` still derives 0.75, the
instant-delivery frontier still lands near it, and this one does not.

Note also what is **not** happening here: the top two rows saturate at a fill rate of
1.0000, but the bottom four slope. That is the difference between this and the flattened
frontier described under *what didn't work* below, where every row was 1.0000 and the
chart had nothing in it. Recorded in
[ADR 0010](decisions/0010-the-pipeline-is-opt-in-and-the-critical-ratio-does-not-survive-it.md).

---

## What didn't work

*Two things, both kept because a dead end costs a day whether or not it is written down.*

### The quantile models were not calibrated, and are now only half fixed

`python -m stockout calibration` on the newest fold, trading days only:

| Nominal quantile | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| `gbm_quantile` covers | 0.357 | 0.564 | 0.607 | 0.721 | 0.843 | 0.893 |
| `gbm_conformal` covers | 0.414 | 0.650 | 0.693 | 0.779 | 0.886 | 0.929 |

Every level under-covers, and the gap is widest in the middle of the distribution. In
sample the same 0.9 model covers 0.865, so this is generalisation error rather than a
fitting bug: quantile regression fits the spread of the residuals it was shown, and the
spread at a 42-day horizon on held-out data is wider than that.

The consequence was the most interesting number in the repository. **The cost-minimising
service level was 0.90 while the newsvendor critical ratio derived from the cost pair is
0.75** — you had to over-ask by a grid step to get what you asked for.

Split-conformal calibration closes about half of every gap and moves the cost minimum to
0.80, and it improves pinball loss at every level while doing so, so this is not coverage
bought by making the forecast worse. It does not close the gap. After calibration a
nominal 0.9 still delivers 0.779, and no arithmetic performed on a 42-day calibration
window can invent knowledge of the 42-day window that follows it. The raw model is kept
under its own name and `stockout calibration` prints both tables, because the size of the
correction is itself the finding.

Two costs are recorded in full in
[ADR 0009](decisions/0009-conformal-calibration-not-a-recalibrated-loss.md) and one of
them disagrees with the theory outright: **on a two-store draw with 70 calibration rows
the calibration is worse than no calibration**, on both worst-gap (0.107 → 0.171) and
pinball (188.9 → 195.6). Seventy pooled residuals cannot estimate six quantiles. No
threshold is enforced, because a number picked from four draws is a magic constant with a
story attached; `calibration_rows` is printed instead, and levels whose correction rests
on a single observation are named as saturated.

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

The other half of that fix arrived later: `simulate` now takes a `lead_time_days` and
opens a real delivery pipeline, so the `(R, S)` level and the loop that needs it finally
move together. The default is still zero, and the test above still passes untouched.

---

## Machinery, demonstrated — the notebook

Every cell of `notebooks/01_explore.ipynb` runs end to end on the committed sample. None
of it answers anything, and two of the outputs are worth recording anyway because they
show the cells are measuring what they claim to.

**Q1's curve slopes the wrong way.** Seasonal-naive scores WMAPE 0.198 at a 7-day horizon
and 0.143 at 42 — accuracy *improving* with distance, which is the stated **no** for the
hypothesis. On this generator it is an artefact and a legible one: the promotion calendar
alternates weekly, so same-weekday-last-week always lands on the opposite promotion state
at h=7, and lands on the same one at h=14 and h=28. The cell is fine; the data is a
metronome.

**Q4's profile is flat**, at 0.999 on the first day after a promotion and 1.000 thereafter.
That is the correct answer for a generator that applies a constant promotion multiplier
and models no payback at all, and it is the cell's own unit test: a synthetic lift far from
zero would have meant the weekday matching was broken rather than that promotions pull
demand forward.

**Q3's concentration is mild** — the worst 5% of test days carry 15.5% of the absolute
error, the worst 10% carry 26.2%, the worst 20% carry 44.3%. Against a perfectly even
5/10/20 that is concentration, but nothing like the hockey stick real holiday trading
would produce.

**Q5's three policies, on synthetic data, rank the wrong way round.** Pricing
`mean + z·sigma`, the raw quantile and the calibrated quantile at the same derived 0.75
target on store 1 gives total costs of 45,474, 52,864 and 45,252. The textbook normal
policy that ADR 0007 argues against beats the raw quantile policy by 14% and ties the
calibrated one. If that survived to real data it would be a **no** for Q5 and the most
interesting result available — and the reason it cannot be reported as one is the heading
this section sits under. What it does say, unambiguously, is that the raw quantile model's
apparent advantage was calibration all along.

---

Each question below has a cell that runs and produces its number. What none of them has is
data those numbers may be read from. The work left is `python -m stockout fetch`, changing
one line at the top of the notebook, and reading five outputs — not writing five analyses.

## Q1 — Accuracy decay with horizon

*Unanswered.* Cell `q1-code` backtests at 7, 14, 28 and 42 days and saves
`reports/q1_horizon_decay.png`. On synthetic data the curve slopes upward, for the reason
given above.

**Number:**
**Evidence:**
**Verdict:**

## Q2 — Seasonal-naive vs GBM on low-volume stores

*Unanswered.* Cell `q2-code` splits first and measures store volume from the training
window only — segmenting on the whole file would let the test window decide which stores
count as low volume, which is leakage wearing a label rather than a feature. What is
missing is a store population with genuinely varied volume: the generator's four stores
are drawn from one distribution, and four stores cannot populate four quartiles.

**Number:**
**Evidence:**
**Verdict:**

## Q3 — Concentration of error across days

*Unanswered.* Cell `q3-code` ranks test rows by absolute error, plots the cumulative
share against a diagonal, and prints the ten worst days with their promotion and holiday
flags so an answer can say *which* days rather than only how few.

**Number:**
**Evidence:**
**Verdict:**

## Q4 — Post-promotion demand

*Unanswered.* The generator has no post-promotion dip to find — its promotion lift is a
constant multiplier with no payback period — so the null result it produces measures the
generator and nothing else.

Cell `q4-code` had to be rewritten to work at all. The obvious design — compare days in
the wake of a promotion against days that are not — divides by an empty set on any
calendar that promotes every other week, because then every non-promotion day is in some
promotion's wake. It measures the profile against days-since-promotion instead, with each
store's weekday cycle divided out first. That is a real constraint on the question and it
would have gone unnoticed until the Rossmann file landed.

**Number:**
**Evidence:**
**Verdict:**

## Q5 — Newsvendor quantile vs mean-plus-safety-stock

*Unanswered*, and the calibration result above is the reason to expect it to be close.
Cell `q5-code` prices three policies rather than two: `mean + z·sigma`, the raw quantile,
and the calibrated quantile, all at the derived 0.75 target. Sigma is measured on a
held-out tail rather than in sample, so the textbook policy is not handicapped by a
residual spread it has already been shown — the conformal model gives up the same days,
which is what makes the comparison fair rather than rigged.

**Number:**
**Evidence:**
**Verdict:**

## What we did not ask

Anything interesting found while answering the five. Kept separate so that planned
findings and lucky ones are never confused.
