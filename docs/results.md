# Results

> **Every number on this page came from `stockout.data.synth`.** The Rossmann archive
> needs a Kaggle account and an accepted competition-rules page, and this repository's
> rule is that the generator *tests machinery and never supports a finding*. Numbers
> below are therefore evidence that the pipeline runs and produces coherent output. They
> are not evidence about retail, and two of them are interesting precisely because they
> are **negative**.

Every answer needs three things: **a number**, **the chart or table it came from**, and
**a verdict** — held, lost, or inconclusive. "It seems that" is not a verdict.

---

## Machinery, demonstrated

### Regression

Committed synthetic sample, horizon 7, a 28-day held-out window with a 7-day gap, scored
on trading rows only. 2,279 training rows, 92 scored.

| Model | R² | WMAPE | MAE | RMSE | seconds |
|---|---|---|---|---|---|
| `dummy` | −0.003 | 0.229 | 1,703 | 2,261 | 0.1 |
| `linear` | 0.835 | 0.090 | 670 | 918 | 0.0 |
| `ridge` | 0.842 | 0.090 | 669 | 898 | 0.0 |
| `lasso` | 0.840 | 0.090 | 672 | 904 | 0.1 |
| `elastic_net` | 0.779 | 0.095 | 708 | 1,060 | 0.0 |
| `polynomial` | **0.845** | 0.093 | 691 | 888 | 0.2 |
| `decision_tree` | 0.780 | 0.100 | 748 | 1,058 | 0.0 |
| `bagging` | 0.823 | 0.095 | 709 | 949 | 4.1 |
| `random_forest` | 0.823 | 0.095 | 709 | 949 | 0.2 |
| `hist_gradient_boosting` | 0.843 | 0.091 | 674 | 893 | 2.2 |
| `knn` | 0.762 | 0.105 | 778 | 1,101 | 0.1 |
| `svr` | 0.833 | **0.090** | 670 | 923 | 0.2 |

`dummy` scoring −0.003 is the definition of R² working: it predicts the training mean, and
R² *is* the improvement on that.

The interesting result is that **nothing beats `ridge` by enough to matter.** Four models
sit between 0.833 and 0.845, one of them a straight line; `bagging` spends 4.1 seconds to
land below it. On a generator whose structure is multiplicative and whose calendar effects
are constants, a linear model on the right features is the correct answer, and the honest
report is that the ensembles bought nothing. Whether that survives on Rossmann is Q2.

### Classification

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

| Window | Low | Medium | High |
|---|---|---|---|
| training | 0.335 | 0.332 | 0.333 |
| test | 0.392 | 0.358 | 0.250 |

Thirds by construction on the window the cut points were fitted on; visibly skewed 42 days
later. This is why accuracy alone is not reported, and why every classifier is
class-weighted.

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


### The leakage decomposition shows nothing, and the reason is structural

`evaluate/leakage.py` runs four arms against one fixed future window. Three of the four
are indistinguishable:

| arm | protocol | internal | future | optimism |
|---|---|---|---|---|
| honest | time-ordered | 0.930 | 0.933 | −0.003 |
| shuffled split | random | 0.930 | 0.930 | 0.000 |
| preprocessing leak | scaler fitted on all rows | 0.930 | 0.933 | −0.003 |
| future feature | `customers` smuggled in | 1.000 | 1.000 | 0.000 |

**Verdict: inconclusive, and diagnosably so.** Four things were tried before concluding it:

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

So the honest statement is: **a synthetic generator with a stationary process and a
deterministic calendar cannot demonstrate leakage**, and the three flat rows above are a
measurement of the generator, not of the method. The machinery is correct and tested; the
finding waits on Rossmann, where residuals persist, promotions are irregular and the
calendar does not repeat exactly.

The fourth arm is not flat, and it is the one that transfers. `customers` scores **1.000**
— it is in `train.csv`, it correlates with sales at about 0.9, and nobody knows it six
weeks ahead. A model can be perfect on a held-out set and undeployable the same afternoon.
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

Still unanswered. They need Rossmann, for the reason at the top of this page.

| | Question | Status |
|---|---|---|
| Q1 | Does accuracy decay with horizon, and how fast? | machinery ready |
| Q2 | Does seasonal-naive beat a fitted model on low-volume stores? | machinery ready |
| Q3 | How much error comes from a few days? | machinery ready |
| Q4 | Does promotion lift persist or reverse? | machinery ready |
| Q5 | Does the classifier add anything over binning the regressor? | machinery ready |
