# 0005 — WMAPE and MASE; MAPE is excluded

## Situation

MAPE is the default retail forecasting metric and it is a bad one here for three separate
reasons.

1. **It divides by the actual.** Rossmann is full of zeros — every closed day — so MAPE is
   either undefined or has to be patched with an epsilon, and the epsilon quietly decides
   the answer.
2. **It is asymmetric in the wrong direction.** Over-forecasting is bounded at 100% error;
   under-forecasting is unbounded. For a stocking decision that is backwards: a lost sale
   usually costs more than a carried unit, so a metric that punishes over-forecasting
   harder pushes the model the wrong way.
3. **It cannot be aggregated across stores** without implicitly weighting small stores as
   heavily as large ones.

## Decision

Four metrics, each with a job:

| Metric | Job |
|---|---|
| **WMAPE** | Primary. `sum abs(error) / sum abs(actual)` — one division, at the end |
| **MASE** | Skill against seasonal-naive. Below 1 means beaten; nothing else proves that |
| **RMSPE** | Kaggle's official Rossmann metric, so a result stays leaderboard-comparable |
| **Pinball + coverage** | For the quantile models, once they exist |

MASE here is the **same-window** form: it scales by a supplied baseline's MAE over the
evaluation window, not by the in-sample naive MAE of the classical definition. This is
deliberate. The claim worth making is "this beat seasonal-naive on the data it was tested
on", and the number should say exactly that — `seasonal_naive` scores precisely 1.000, and
`test_backtest.py` asserts it as a self-check on the harness.

One property worth knowing, because it is easy to assume the opposite: **WMAPE is
unaffected by including closed days.** A closed day contributes zero to the numerator and
zero to the denominator and cancels exactly. MAE and RMSE, being per-row averages, *are*
flattered by that block of perfect zeros. Both behaviours are asserted in the test suite.

## Cost

**Not directly comparable to published work that quotes MAPE**, which is most of it. A
reader who wants a MAPE has to compute it, and the numbers here will look different from a
tutorial on the same dataset.

**MASE needs a baseline forecast in scope**, so the backtest fits `SeasonalNaive` on every
fold in addition to the model under test. That roughly doubles baseline-fit time, which is
negligible now and would not be with an expensive model.
