# 0012 — Measure coverage per group always; correct per group only when the rows allow it

## Situation

ADR 0009 recorded a cost it could not see:

> **The correction is marginal, not conditional.** Coverage is corrected on average across
> the pooled rows. A particular store, or December, can still be badly covered, and this
> layer will not notice. Conditional coverage needs either per-group calibration — which
> runs straight into the row counts below — or a Mondrian scheme, and neither is here.

"Will not notice" is the operative phrase. A marginal coverage table is an average, and an
average over groups is exactly the statistic that cannot report a bad group. The worst
case is not subtle: two stores, one covered on every day and one covered on six days in
ten, average to precisely the nominal 0.8 and report a gap of zero.

So the question is not whether the marginal number was wrong. It was right, and it was
right about something nobody stocks: inventory is held per store, and a service level that
holds on average across four shops is not a promise any one of them can keep.

## Decision

Two separate things, because measuring a defect and correcting it are different jobs and
the second is much more expensive than the first.

**`metrics.coverage_by_segment` measures, and it is free.** One coverage row per group per
level, with the row count each number rests on, rendered by
`report.conditional_coverage_to_markdown` as a grid that names its own worst cell.
`stockout calibration --by store` and `--by month` print it beside the marginal table. This
is on demand rather than always, and it changes no existing number.

**`ConformalQuantileForecaster(group_by=...)` corrects, and it is not free.** A separate
pooled offset per group — Mondrian conformal. It is off by default.

**The row floor is derived, not chosen.** Splitting the residual pool means every group is
estimated from a fraction of the rows, and ADR 0009 already measured what happens below a
floor: at 70 pooled rows the calibration was *worse than none*. Rather than pick a
threshold — "a magic constant wearing a justification", in that ADR's words — the floor is
`conformity.min_rows_for`, the smallest row count at which the strictest level in the grid
stops saturating. It is 3 rows for a grid topping out at 0.5, 19 for a 0.9, and **199 for a
0.99**. Asking for a higher service level per group is expensive by arithmetic rather than
by opinion.

Groups below the floor get no offset and take the marginal one, and
`pooled_fallback_groups` names them. A group the calibration window never held — a shop
that opened since — takes the marginal offset too, and `unseen_groups` names it. Degrading
from a conditional correction to a marginal one is the right fallback and must not be a
silent one.

**Only `store` can be corrected, and `--calibrate-by` offers nothing else.** This is a real
asymmetry rather than an unfinished feature. A Mondrian group has to exist in both the
calibration window and the window being predicted. Stores do. Months do not: the
calibration window is the 42 days *before* the test window, so it contains none of the
months whose coverage is in question, and there is no December residual to correct December
with. December can be measured and cannot be calibrated, and `--by month` versus
`--calibrate-by store` is that fact in the argument surface.

## Consequence — the marginal table was understating the worst store by two thirds

`python -m stockout calibration --by store`, newest fold, four synthetic stores, 35 trading
rows each. Coverage gap per store:

**`gbm_quantile`** (raw)

| store | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| 1 | −0.129 | −0.264 | **−0.314** | −0.186 | −0.179 | −0.161 |
| 2 | −0.157 | −0.036 | −0.029 | −0.071 | +0.021 | +0.010 |
| 3 | −0.214 | −0.293 | −0.229 | −0.300 | −0.264 | −0.190 |
| 4 | −0.071 | −0.150 | −0.200 | −0.157 | −0.007 | −0.047 |

**`gbm_conformal`** (marginally calibrated)

| store | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| 1 | −0.071 | **−0.207** | −0.171 | −0.157 | −0.150 | −0.104 |
| 2 | −0.157 | +0.050 | +0.000 | −0.071 | +0.021 | +0.010 |
| 3 | −0.157 | −0.150 | −0.171 | −0.186 | −0.121 | −0.104 |
| 4 | +0.043 | −0.093 | −0.086 | −0.071 | −0.007 | −0.047 |

Three things fall out, and the third is the one worth keeping.

**The marginal table understates the worst store by about two thirds.** The raw model's
worst marginal gap is −0.193 at the 0.80; store 1's gap at that level is −0.314, worse by a
factor of 1.6. After calibration the worst marginal gap is −0.121 and the worst store gap
is −0.207, again a factor of 1.7. Both numbers were published, and only the smaller one.

**A pooled offset has to be wrong in two directions at once.** After calibration store 2
*over*-covers at 0.75, 0.95 and 0.99 while store 1 under-covers at every level. One
additive correction cannot move a shop up and a shop down, and this is what that
impossibility looks like in a table.

**Marginal calibration nonetheless improved the conditional picture, and for a reason
already in the code.** At the 0.90 the per-store spread narrows from 0.229 raw to 0.115
calibrated. That is not luck: ADR 0009's conformity score is divided by the model's own
prediction for the row, so the correction is *relative* and already scales with store
level. A raw offset in currency units would have widened the spread. The scaled score was
justified on heteroscedasticity grounds and turns out to have bought a share of conditional
validity as well.

## Cost

**On this data the correction declines to act, and that is the honest result.**
`--calibrate-by store` with the default 0.99 grid needs 199 trading rows per store; four
synthetic stores hold 35 each. Every group falls back to the pooled offset and the command
says so — `0 group(s) cleared it; 4 fell back`. The feature exists, is tested, and reports
that this dataset cannot support it. Rossmann is worse per store, not better: 1,115 stores
against a 42-day calibration window is about 36 trading rows each. Per-store Mondrian at a
0.99 service level needs either a much longer calibration window or many fewer groups —
coarser groups (store cluster, volume quartile, region) are the way this becomes usable,
and no clustering exists here to build them from.

**The diagnostic is itself thin, and says so rather than hiding it.** A gap measured on 35
rows moves in steps of 0.029 and carries a standard error near 0.05, so the difference
between store 2 and store 4 above is noise and the difference between store 2 and store 1
is not. `rows` is printed in every conditional table for exactly this reason; a conditional
grid without its row counts invites the over-reading it was built to prevent.

**Two coverage numbers now exist for one model**, and they disagree by design. Anyone
quoting the marginal one after this ADR is quoting the smaller of two numbers they were
shown.

**Month-shaped miscalibration remains measurable and uncorrectable.** ADR 0009's "a
particular store, or December" is now half addressed. The store half can be measured and
in principle corrected; the December half can only be measured, and the reason is the same
distribution shift that leaves the marginal gap at −0.121 after calibration. No arithmetic
on the 42 days before a window reveals the seasonality inside it.
