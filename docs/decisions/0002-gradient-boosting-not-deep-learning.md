# 0002 — Gradient boosting, not a sequence model

## Situation

"Demand forecasting" invites an LSTM, a Temporal Fusion Transformer, or whatever is
current. The Kaggle competition this data comes from was won by gradient-boosted trees
with careful feature engineering, and the M5 competition that followed was won the same
way.

## Decision

The model ladder stops at LightGBM: `naive_last` → `seasonal_naive` → `moving_average` →
LightGBM point (`objective="tweedie"`) → LightGBM quantile.

No LSTM, no Transformer, no Prophet.

The reasoning is that on tabular retail data with strong calendar structure, boosted trees
train in seconds, handle the categorical and count features natively, and — decisively —
can be explained under questioning. A sequence model here would add a GPU dependency,
several days of build time, and a result that is most likely slightly worse.

The `tweedie` objective is the non-obvious part and is the defensible one. Sales are
non-negative, continuous above zero, and carry a point mass at zero from closures. That is
a compound Poisson-gamma distribution, which is what tweedie fits. Squared error on the
raw target over-weights the high tail; squared error on `log1p` biases the back-transformed
mean low. Tweedie is tried first and the comparison is reported rather than asserted.

## Cost

**No cross-series learning.** A single global neural model can share structure across
stores, which genuinely helps stores with short histories. The per-store baselines here
cannot, and question Q2 in [questions.md](../questions.md) exists to measure how much that
costs on the bottom volume quartile.

**A ceiling.** If the answer to Q2 is that low-volume stores are badly served, the fix is
a hierarchical or global model, and this decision has to be revisited rather than defended.
