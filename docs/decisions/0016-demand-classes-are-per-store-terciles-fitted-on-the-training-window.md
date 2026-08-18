# 0016 — Demand classes are per-store terciles, fitted on the training window

## Situation

The classification half of this package predicts whether a store-day is **Low**, **Medium**
or **High** demand. That phrase means nothing until somebody fixes the cut points, and
there are three separate ways to get it wrong.

**Global cut points measure the wrong thing.** One pair of thresholds across all stores
puts almost every day of a busy shop in *High* and almost every day of a quiet one in
*Low*. A classifier trained against those labels scores well by learning store identity,
which it can read straight off the `store` column, and knows nothing about demand.

**Cut points fitted on all the data are leakage.** A tercile is a statistic of a sample.
Computed over the whole frame it has been computed partly from the test window, which has
then voted on its own labels. This is the same leak a `StandardScaler` fitted outside a
`Pipeline` causes, and it is harder to see because nothing about it looks like a model.

**Closed days have no class.** A shut store sells zero. Binning that zero puts the entire
closure mass in the bottom bin, so *Low* comes to mean *shut* — a question nobody asked
and every model can already answer from the `open` column.

## Decision

Cut points are the **1/3 and 2/3 quantiles of each store's own trading-day sales**, fitted
on the training rows alone, with a pooled pair as fallback for a store the training window
never saw. Closed days get `pd.NA` rather than a class, and are dropped before fitting.

`evaluate/comparison.py` refits the thresholds on its training slice before scoring
anything, even though `dataset.prepare` has already labelled the frame. That second fit
costs one pass over a column and removes the objection.

The alternative is kept and measured rather than merely argued against.
`targets.absolute_thresholds` builds the global version, and on the committed sample —
28-day holdout, 7-day gap — it produces this:

| labels | dummy | logistic | hist_gradient_boosting |
|---|---|---|---|
| per-store terciles | 0.156 | 0.691 | 0.703 |
| one global pair | 0.129 | **0.810** | 0.736 |

*(macro-F1)*

Logistic regression gains **0.119 macro-F1** from the switch, which is a larger gap than
separates any two models in the entire regression table. It has not become a better model.
It has been handed a shortcut, and the linear model takes it hardest because store
identity is exactly the kind of thing a weighted sum of one-hot columns can encode
perfectly.

The training-window class balance is 33/33/33 by construction; on this test window it
drifts to 30/34/36 under per-store labels and to 24/46/30 under global ones. That drift is
why macro-F1 leads the table and accuracy follows it — a model can drop the smallest class
entirely, score respectably on accuracy, and be useless at the job.

## Cost

**Three classes with a 33% floor make every accuracy look mediocre.** A published 0.70 on
this problem is a real result and reads like a poor one. `DummyClassifier` is the first row
of every classification table for that reason, and its 0.30 is what makes the 0.70 legible.

**A store with no training rows takes the pooled cut points**, which is exactly the
global-threshold mistake applied to a minority of rows. Roughly a sixth of Rossmann's
stores vanish for a refurbishment quarter, so this is not a defensive branch for a case
that cannot happen. Those rows are labelled by a rule the ADR argues against; the number
of stores it affects is reported by `DemandThresholds.stores`.

**The thresholds must be shipped with the model.** A predicted number is a number; turning
it into a label needs that store's cut points, which are a property of the training window
rather than of the estimator. `persistence.py` carries them for this reason, and an
artifact without them cannot answer the classification half at all.

**The labels are not the business's labels.** No retailer defines a busy day as the top
third of that shop's own history. A real definition would come from a planner and would
probably not be a tercile at all.
