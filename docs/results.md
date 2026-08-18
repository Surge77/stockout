# Results

> **Every number on this page came from `stockout.data.synth`.** The Rossmann archive
> needs a Kaggle account and an accepted competition-rules page, and this repository's
> rule is that the generator *tests machinery and never supports a finding*. Numbers
> below are therefore evidence that the pipeline runs and produces coherent output. They
> are not evidence about retail, and several of them are interesting precisely because
> they are **negative**.

Every answer needs three things: **a number**, **the chart or table it came from**, and
**a verdict** — held, lost, or inconclusive. "It seems that" is not a verdict.

Every table on this page is reproducible from the committed sample by the command named
above it. If one is not, that is a bug in this file rather than a nuance.

---

## Machinery, demonstrated

### Regression

`python -m stockout compare`. Committed synthetic sample, horizon 7, a 28-day held-out
window with a 7-day gap, scored on trading rows only. 2,279 training rows, 92 scored.

| Model | R² | WMAPE | MAE | RMSE | seconds |
|---|---|---|---|---|---|
| `dummy` | −0.003 | 0.229 | 1,703 | 2,261 | 0.1 |
| `linear` | 0.835 | 0.090 | 669 | 918 | 0.0 |
| `ridge` | 0.842 | **0.090** | 669 | 898 | 0.0 |
| `lasso` | 0.840 | 0.090 | 672 | 904 | 0.1 |
| `elastic_net` | 0.779 | 0.095 | 708 | 1,060 | 0.0 |
| `polynomial` | **0.845** | 0.093 | 691 | 888 | 0.3 |
| `decision_tree` | 0.780 | 0.100 | 748 | 1,058 | 0.0 |
| `bagging` | 0.823 | 0.095 | 709 | 949 | 7.7 |
| `random_forest` | 0.823 | 0.095 | 709 | 949 | 0.2 |
| `hist_gradient_boosting` | 0.843 | 0.091 | 674 | 893 | 1.9 |
| `knn` | 0.762 | 0.105 | 778 | 1,101 | 0.1 |
| `svr` | 0.833 | 0.090 | 670 | 923 | 0.1 |

The `seconds` column is one laptop's, and it is the only column on this page that will not
reproduce. It is reported anyway: a model that scores 0.843 in two seconds and one that
scores 0.823 in eight are not the same result, and a table that omits the second axis is
quietly recommending the wrong one.

`dummy` scoring −0.003 is the definition of R² working: it predicts the training mean, and
R² *is* the improvement on that.

The interesting result is that **nothing beats `ridge` by enough to matter.** Four models
sit between 0.833 and 0.845, one of them a straight line, and at four decimal places the
straight line and the winner are the same number — WMAPE 0.0899 each. `bagging` spends 7.7
seconds to land below both. On a generator whose structure is multiplicative and whose
calendar effects are constants, a linear model on the right features is the correct answer,
and the honest report is that the ensembles bought nothing. Whether that survives on
Rossmann is unanswered.

### Classification

`python -m stockout compare --task classification`, same window and same rows.

| Model | accuracy | macro-F1 | adjacent | recall Low | recall Med | recall High | seconds |
|---|---|---|---|---|---|---|---|
| `dummy` | 0.304 | 0.156 | 0.641 | 1.000 | 0.000 | 0.000 | 0.0 |
| `logistic` | 0.685 | 0.691 | 1.000 | 0.786 | 0.613 | 0.667 | 0.2 |
| `decision_tree` | 0.576 | 0.579 | 0.989 | 0.821 | 0.516 | 0.424 | 0.1 |
| `bagging` | 0.565 | 0.567 | 1.000 | 0.821 | 0.548 | 0.364 | 0.4 |
| `random_forest` | 0.630 | 0.638 | 1.000 | 0.786 | 0.581 | 0.545 | 0.2 |
| `hist_gradient_boosting` | **0.696** | **0.703** | 1.000 | 0.821 | 0.677 | 0.606 | 1.8 |
| `knn` | 0.620 | 0.624 | 0.967 | 0.786 | 0.581 | 0.515 | 0.1 |
| `svc` | 0.674 | 0.677 | 1.000 | 0.893 | 0.677 | 0.485 | 0.2 |
| `linear_svc` | 0.674 | 0.677 | 1.000 | 0.857 | 0.548 | 0.636 | 0.2 |

`dummy` at 0.304 accuracy and 0.156 macro-F1 is the floor, and the gap between those two
numbers is the argument for reporting both: predicting Low every time is right 30% of the
time and has a recall of 1.000 on Low and 0.000 on everything else. Accuracy alone would
call that a third of a model. Macro-F1 calls it a sixth.

Every real model scores `adjacent` at or near 1.000 — when they are wrong they are wrong
by one class, never by two. That is worth knowing and no standard metric reports it.

### Class balance drifts, and that is the finding

Cut points fitted on the training window, then applied to both.

| Window | Low | Medium | High |
|---|---|---|---|
| training | 0.334 | 0.333 | 0.333 |
| test | 0.304 | 0.337 | 0.359 |

Thirds by construction on the window the cut points were fitted on; visibly skewed 28 days
later, and skewed *towards High* — the classifier is asked about a busier month than the
one it learned "busy" from. This is why accuracy alone is not reported, and why every
classifier is class-weighted.

### The labels could have looked much better than they are

`targets.absolute_thresholds` builds the version this project rejects: one pair of cut
points for every store rather than per-store terciles. Same models, same rows, same window:

| labels | `dummy` | `logistic` | `hist_gradient_boosting` |
|---|---|---|---|
| per-store terciles | 0.156 | 0.691 | 0.703 |
| one global pair | 0.129 | **0.810** | 0.736 |

*(macro-F1)*

Logistic regression gains 0.119 from the switch — a larger gap than separates any two
models in the whole regression table. It has not become a better model; it has been handed
store identity as a shortcut, and a linear model takes that hardest because one-hot store
identity is exactly what a weighted sum encodes perfectly.

**Verdict: the rejected alternative scores better and measures less.** Kept and measured
rather than argued against, because a claim is stronger with its alternative beside it —
[ADR 0016](decisions/0016-demand-classes-are-per-store-terciles-fitted-on-the-training-window.md).

---

## What didn't work

### The regressor was reading its own answer

`demand_class_code` is a tercile of `sales`. `feature_columns` admitted it because it is
numeric and was not the *current* task's target, so every regression model in this project
was handed a three-way summary of the number it was predicting.

Found by the serving path rather than by a test: `predict` builds a future row with no
label on it, the fitted pipeline asked for a column that was not there, and sklearn said
`columns are missing: {'demand_class_code'}`. A leak that improves a score is invisible; a
leak that breaks an unrelated code path is not, which is the only reason this one surfaced.

What it was worth:

| Model | R² leaked | R² clean | WMAPE leaked | WMAPE clean |
|---|---|---|---|---|
| `ridge` | 0.887 | 0.842 | 0.068 | 0.090 |
| `hist_gradient_boosting` | 0.900 | 0.843 | 0.067 | 0.091 |

About 0.05 of R² and a quarter of WMAPE. Every regression figure published before this
correction was inflated by it.

The fix is at the schema level rather than at the call site. `schemas.TARGET_COLUMNS` names
all three of `sales`, `demand_class` and `demand_class_code`, and `FEATURE_DENYLIST` refuses
the set — because excluding "the target of this task" is not enough when two targets encode
each other.


### The leakage decomposition shows almost nothing, and the reason is structural

`evaluate/leakage.py` runs four arms against one fixed future window. `python -m stockout
leakage`, 28-day window with a 7-day gap, R²:

| arm | protocol | internal | future | optimism |
|---|---|---|---|---|
| honest | time-ordered | 0.9013 | 0.8447 | 0.0566 |
| shuffled split | random | 0.9131 | 0.8413 | 0.0718 |
| preprocessing leak | scaler and imputer fitted on all rows | 0.9014 | 0.8447 | 0.0567 |
| future feature | `customers` smuggled in | 0.9984 | 0.9988 | **−0.0004** |

**Verdict: inconclusive for three arms, and diagnosably so.** The shuffled split is
optimistic by 0.0718 against the honest arm's 0.0566 — a gap of 0.015 R², which points the
right way and is far smaller than this project once implied. The preprocessing leak differs
in the fourth decimal. Four things were tried before concluding it:

1. **Ridge cannot memorise**, so a shuffled split buys it nothing. Rerun with
   `RandomForest(min_samples_leaf=2)` and with `KNN(k=3)` — the memorisation case in the
   limit. Optimism under a random split: −0.009 and −0.010. Still nothing.
2. **The lag guard might already be closing the channel.** Bypassed it, adding `sales_lag_1`
   to a horizon-7 model. Optimism: −0.005. Still nothing.
3. **The residual might be too well behaved.** It was: the generator's noise was i.i.d., so
   *no* neighbouring observation carried information about any other. Every leak works by
   letting a model see an informative neighbour, and there were none to see. The generator
   grew a fifth structure — AR(1) residuals, correlation 0.62 at one day and 0.17 at seven —
   for this reason. Documented in `data/synth.py`.
4. **Even with persistent residuals it does not appear**, because the evaluation hands
   `lag_1` to the future window as well. A leak only costs you when the feature is absent
   at serving time, and an offline experiment that supplies it everywhere cannot show that.

So the honest statement is: **a synthetic generator with a deterministic calendar can
barely demonstrate leakage**, and the near-flat rows above are mostly a measurement of the
generator rather than of the method. The machinery is correct and tested; the finding waits
on Rossmann, where residuals persist, promotions are irregular and the calendar does not
repeat exactly.

The fourth arm is not flat, and it is the one that transfers. `customers` scores **0.998**
on a genuinely held-out window — it is in `train.csv`, it correlates with sales at about
0.9, and nobody knows it six weeks ahead. Its optimism is *negative*, which is the sharpest
form of the point: this protocol is not lying about its own error at all, and the model is
still undeployable the same afternoon. A high score on a held-out set is not evidence that
a model can be used.

Note also what it took to run that arm at all: `FEATURE_DENYLIST` refuses the column, so
the experiment has to **rename it** to get past the guard. Needing to lie to a guard in
order to demonstrate what it prevents is the strongest evidence available that it is
load-bearing.

### Sunday is never in the training data

Models are fitted on trading rows, and in this generator every Sunday is closed. The
one-hot encoder therefore learned `day_of_week` categories 1–6, and every Sunday row at
predict time arrived as an unseen category. Caught by a `UserWarning` promoted to an error
by `filterwarnings = ["error"]`; fixed by predicting only on the rows the model was fitted
for and filling the rest. Real Rossmann has Sunday-opening stores, so the same code will
behave differently there — which is worth knowing before it surprises somebody.

### Reproducibility is not bit-for-bit

A `RandomForest` with `n_jobs=-1` averages its trees across threads, and floating-point
addition is not associative. Two runs with the same seed differ by about 4×10⁻¹⁶ relative.
The test asserts `allclose` rather than equality, with a partner test on a different seed
so the loosened assertion cannot pass vacuously.

---

## The five questions

Answered in [`notebooks/01_explore.ipynb`](../notebooks/01_explore.ipynb), on the
generator. **Read every verdict below as a statement about `stockout.data.synth`**, for the
reason at the top of this page — and note that three of the five lost or came out flat,
which is what committing the questions before the analysis is for.

| | Question | Verdict |
|---|---|---|
| Q1 | Does accuracy decay with horizon? | **Lost** |
| Q2 | Does the baseline beat a fitted model on quiet stores? | **Lost** |
| Q3 | Is the error concentrated in a few days? | **Held** |
| Q4 | Does the promotion lift persist, or reverse? | **Held, weakly** |
| Q5 | Does the classifier beat binning the regressor? | **Held** |

### Q1 — accuracy does not decay with horizon here

`ridge` and `seasonal_naive`, five rolling-origin folds, one prepared frame per horizon
because the lag a model may read is defined by the horizon it forecasts at:

| horizon | `ridge` WMAPE | `seasonal_naive` WMAPE |
|---|---|---|
| 7 | 0.0849 | 0.2004 |
| 14 | 0.0840 | 0.1424 |
| 28 | 0.0835 | 0.1449 |
| 42 | 0.0875 | 0.1462 |

**Verdict: lost, on the stated condition.** The question named "WMAPE flat across horizons"
as the failure case and that is what happened — 0.0849 to 0.0835 to 0.0875, non-monotonic
and inside a range of 0.004. Forecasting six weeks out is no harder here than forecasting
one, which is only possible because the generator's structure is almost entirely calendar:
promotions repeat on a fixed cycle and the weekday pattern is deterministic, so a model with
calendar features barely needs recent history. On Rossmann the recent history matters and
this curve should slope.

The baseline's own numbers move more than the model's, and not monotonically either. That
is a fold-layout artefact rather than a finding: changing the horizon changes which days
land in which fold.

### Q2 — the fitted model wins on every store, quiet ones included

MASE against seasonal-naive on the held-out window, stores ordered quietest first. Two
fitted models, because "the fitted model" is ambiguous: **pooled** trains on every store and
is what you would deploy; **per store** trains on that store's rows alone.

| store | mean sales | train rows | pooled | per store |
|---|---|---|---|---|
| 4 | 5,699 | 589 | 0.499 | 0.462 |
| 3 | 5,815 | 589 | 0.601 | 0.497 |
| 2 | 7,944 | 512 | 0.439 | 0.482 |
| 1 | 9,905 | 589 | 0.604 | 0.685 |

**Verdict: lost.** The question named "the fitted model wins uniformly across every store"
as the failure case. It does, both ways, and there is no relationship between store volume
and MASE — the worst pooled result is the *busiest* store. With four stores of roughly 589
training rows each there is no low-volume regime to find, so this is a null result on data
that cannot produce a positive one rather than evidence against the hypothesis.

One thing in that table is worth keeping: pooling and specialising trade places. The two
quiet stores do better with their own model, the busiest does much worse with one. That is
the shape the question was looking for, in the opposite direction, on a sample far too
small to conclude anything from.

### Q3 — the error is concentrated, and it is all December

`ridge` on the held-out window, test rows ranked by absolute error:

| worst share of days | share of total error |
|---|---|
| 5% | 18.9% |
| 10% | 29.2% |
| 20% | 46.6% |
| 50% | 80.9% |

The five worst days:

| date | store | actual | predicted | error | promo | school holiday |
|---|---|---|---|---|---|---|
| 2014-12-08 | 1 | 16,617 | 12,612 | 4,005 | yes | no |
| 2014-12-22 | 3 | 10,853 | 8,722 | 2,131 | yes | yes |
| 2014-12-27 | 1 | 12,000 | 10,116 | 1,884 | no | yes |
| 2014-12-20 | 3 | 7,875 | 6,061 | 1,814 | no | yes |
| 2014-12-05 | 1 | 12,302 | 10,508 | 1,794 | yes | no |

**Verdict: held.** The worst tenth of days carries three times its share. Every one of the
five worst is in December and four of five carry a promotion or a school holiday, which is
what the question predicted — the concentration is on event boundaries rather than spread
evenly. The implication it was asked for follows: effort belongs in an event model, not in
more features for ordinary days.

The caveat is that the held-out window *is* December. A 28-day holdout at the end of this
calendar cannot separate "December is hard" from "the last month is hard", and a rolling
version of this chart would.

### Q4 — the lift reverses, but only just

Sales against a baseline matched on **store and weekday**, because Saturday outsells Tuesday
by more than any promotion does:

| trading days after the promotion ended | lift |
|---|---|
| 1 | −3.2% |
| 2 | −1.8% |
| 3 | +1.6% |
| 4 | −0.5% |
| 5 | −3.9% |
| 6 | −3.1% |

While the promotion runs: **+24.5%**. Averaged over the wake: **−1.8%**.

**Verdict: held, weakly.** Five of six days sit below the matched baseline, so demand is
pulled forward rather than created — but a −1.8% dip against a +24.5% lift does not recover
anything like the demand borrowed, so on this generator promotions mostly *do* grow demand.
The stated failure condition was "post-promotion sales at or above baseline", and they are
below it, so the hypothesis survives on the letter and barely on the substance.

The wake is six days rather than seven because this generator's promotion cycle never leaves
seven clear trading days between one promotion and the next. A first attempt at this cell
counted straight through the following promotion and reported +22% on days 7 and 8 — a
persistence finding that was entirely the *next* cycle's lift. The counter now resets on any
promotion day, which is the only reason the number above is not three times too high.

### Q5 — the classifier is worth about 0.04 macro-F1

Same estimator (`hist_gradient_boosting`), same rows, same window. One asked to predict the
class; one asked to predict `sales`, whose output is then binned against the same per-store
cut points.

| | classifier | regress then bin |
|---|---|---|
| accuracy | 0.696 | 0.663 |
| macro-F1 | **0.703** | 0.661 |
| adjacent | 1.000 | 1.000 |
| recall Low | 0.821 | 0.571 |
| recall Medium | 0.677 | 0.581 |
| recall High | 0.606 | **0.818** |

**Verdict: held.** The classifier wins by 0.042 macro-F1, so the classification half is not
answering a question the regression half already answered.

The interesting part is not the total but the trade. Binning the regression is dramatically
*better* on High (0.818 against 0.606) and dramatically worse on Low (0.571 against 0.821),
because a squared-error fit is pulled towards the busy days that dominate the loss and its
predictions therefore sit high. The classifier, class-weighted, spreads its errors evenly.
Which of those is preferable is a business question — a missed High is an empty shelf — and
macro-F1 cannot express it, which is the same limitation `adjacent` exists to patch.
