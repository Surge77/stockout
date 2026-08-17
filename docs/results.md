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

Synthetic sample, 10 stores × 900 days, horizon 7, a 42-day held-out window with a 7-day
gap. Scored on trading rows only.

### Regression

| Model | rows fitted | R² | WMAPE | seconds |
|---|---|---|---|---|
| `dummy` | 3,277 | −0.000 | 0.296 | 0.1 |
| `linear` | 3,277 | 0.949 | 0.057 | 0.0 |
| `ridge` | 3,277 | 0.949 | 0.057 | 0.0 |
| `lasso` | 3,277 | 0.947 | 0.058 | 0.1 |
| `elastic_net` | 3,277 | 0.917 | 0.067 | 0.0 |
| `polynomial` | 3,277 | 0.954 | 0.054 | 0.2 |
| `decision_tree` | 3,277 | 0.936 | 0.065 | 0.1 |
| `bagging` | 3,277 | 0.944 | 0.061 | 3.0 |
| `random_forest` | 3,277 | 0.944 | 0.061 | 0.3 |
| `hist_gradient_boosting` | 3,277 | **0.958** | **0.054** | 2.4 |
| `knn` | 3,277 | 0.904 | 0.080 | 0.2 |
| `svr` | 3,277 | 0.949 | 0.054 | 0.2 |

`dummy` scoring −0.000 is the definition of R² working: it predicts the training mean, and
R² *is* the improvement on that. Everything else should be read as "and what did the extra
complexity buy over 0.949", which for most of the table is nothing.

### Classification

| Model | rows fitted | accuracy | macro-F1 | seconds |
|---|---|---|---|---|
| `dummy` | 3,277 | 0.338 | 0.168 | 0.0 |
| `logistic` | 3,277 | 0.738 | 0.742 | 0.3 |
| `decision_tree` | 3,277 | 0.705 | 0.703 | 0.1 |
| `bagging` | 3,277 | 0.686 | 0.681 | 0.5 |
| `random_forest` | 3,277 | 0.686 | 0.681 | 0.3 |
| `hist_gradient_boosting` | 3,277 | 0.700 | 0.708 | 1.9 |
| `knn` | 3,277 | 0.695 | 0.696 | 0.1 |
| `svc` | 3,277 | **0.743** | **0.743** | 0.5 |
| `linear_svc` | 3,277 | 0.719 | 0.719 | 0.5 |

`dummy` at 0.338 is the three-class floor, and it is the reason 0.743 can be read at all.
Note that the linear models win both halves here — on a generator whose structure is
multiplicative and whose calendar effects are constants, that is what should happen.

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
