# Results

> **Nothing here yet.** The five questions in [questions.md](questions.md) are unanswered
> until the analysis in [`notebooks/01_explore.ipynb`](../notebooks/01_explore.ipynb) is
> done. This file is a template, not an omission.

Every answer needs three things: **a number**, **the chart or table it came from**, and
**a verdict on the hypothesis** — held, lost, or inconclusive. "It seems that" is not a
verdict.

---

## Baseline — what is already known

Established by `python -m stockout backtest`, on the committed synthetic sample
(4 stores, 730 days, 5 rolling-origin folds, 42-day horizon). These are not findings; they
are the floor everything else is measured against.

| Model | Mean WMAPE | MASE | Verdict |
|---|---|---|---|
| `seasonal_naive` | 0.1489 | 1.000 | The baseline, by definition |
| `moving_average` | 0.1606 | 1.080 | 8.0% worse |
| `naive_last` | 0.1695 | 1.144 | 14.4% worse |

The gap between `seasonal_naive` and `naive_last` is the value of knowing what day of the
week it is. On real Rossmann data it should be larger.

---

## Q1 — Accuracy decay with horizon

*Unanswered.*

**Number:**
**Evidence:**
**Verdict:**

## Q2 — Seasonal-naive vs GBM on low-volume stores

*Unanswered.*

**Number:**
**Evidence:**
**Verdict:**

## Q3 — Concentration of error across days

*Unanswered.*

**Number:**
**Evidence:**
**Verdict:**

## Q4 — Post-promotion demand

*Unanswered.*

**Number:**
**Evidence:**
**Verdict:**

## Q5 — Newsvendor quantile vs mean-plus-safety-stock

*Unanswered.*

**Number:**
**Evidence:**
**Verdict:**

---

## What didn't work

*Empty on purpose.* Fill it in as things fail — a dead end costs a day whether or not it
is written down, and writing it down is the only way that day buys anything.

Worth recording here: features that added nothing, a model that lost to the baseline,
a metric that turned out to measure the wrong thing, and any assumption that the data
contradicted.

## What we did not ask

Anything interesting found while answering the five. Kept separate so that planned
findings and lucky ones are never confused.
