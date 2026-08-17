# 0011 — Stock in transit is charged for, at the shelf rate until told otherwise

## Situation

ADR 0010 opened a delivery pipeline and wrote down what it left undone:

> **The delivery pipeline is opt-in, and stock in transit is free.** Nothing is charged
> for goods on a lorry, so the model prefers a long pipeline to a full shelf in a way a
> financed business would not.

That was a real defect and not a rounding one. On store 1 with a seven-day lead time the
shelf holds a mean of 5,735 and the pipeline holds 44,819 — **the lorry carries roughly
eight times what the shop does**, and the simulator was pricing one of them. Any
comparison between an instant-delivery policy and a pipeline policy was therefore a
comparison in which one side got most of its inventory for nothing.

## Decision

`simulate` takes a `transit_holding_cost` and charges it, per unit per day, on whatever is
still on the water at the end of each day — the same instant the shelf is charged on.
`SimulationResult.transit_cost` and the frontier's `transit_cost` column report it
separately, and `total_cost` is now holding plus transit plus shortage.

**The default is the on-hand rate, not zero.** Three candidates were available and two are
worse:

- **Zero** is the status quo and is wrong: the capital is committed, the invoice exists,
  and the goods earn nothing until they land.
- **A fraction of the on-hand rate** is defensible in theory — on-hand also buys warehouse
  space, insurance and shrinkage, and a lorry buys none of those, so the true transit rate
  is genuinely lower. But any particular fraction would be a number chosen to look
  reasonable, which is the magic constant ADR 0009 already refused once.
- **The on-hand rate**, which is an explicit upper bound rather than an estimate. It
  cannot understate the financing cost, it needs no justification beyond "capital is
  capital", and it is one keyword away from the operator's real number.

`transit_holding_cost=0.0` recovers the old behaviour exactly, which is the FOB-destination
case: the goods belong to the supplier until they arrive, so nothing is owed for carrying
them. `--transit-holding-cost` exposes both on the command line.

**It is a separate column and not folded into `holding_cost`.** A single blended figure
could not be audited against the `mean_on_order` sitting next to it, and the whole reason
this defect survived a release is that nobody could see what the pipeline was contributing.

## Consequence — the charge is large and changes nothing

`python -m stockout frontier --model gbm_conformal --lead-time 7`, store 1, newest fold:

| quantile | holding | transit | shortage | total | mean on hand | mean on order |
|---|---|---|---|---|---|---|
| 0.50 | 240,884 | 1,882,379 | 30,732 | **2,153,995** | 5,735 | 44,819 |
| 0.75 | 309,002 | 1,925,683 | 4,975 | 2,239,661 | 7,357 | 45,850 |
| 0.80 | 326,746 | 1,928,135 | 2,615 | 2,257,496 | 7,780 | 45,908 |
| 0.90 | 378,419 | 1,930,876 | 0 | 2,309,295 | 9,010 | 45,973 |
| 0.95 | 451,402 | 1,930,730 | 0 | 2,382,132 | 10,748 | 45,970 |
| 0.99 | 582,765 | 1,913,350 | 0 | 2,496,114 | 13,875 | 45,556 |

Total cost rises about **eightfold** — 271,616 to 2,153,995 at the cheapest level — and the
cheapest level is still 0.50. Both halves of that are the finding.

**Why the ranking does not move: the pipeline is throughput, not policy.** In steady state
the quantity in transit is the arrival rate times the lead time, and the arrival rate is
demand — which the service level does not change. So `transit_cost` spans 1,882,379 to
1,930,876 across the whole grid, a range of **2.6%**, while `holding_cost` more than
doubles over the same rows. The charge is very nearly a constant added to every policy, and
adding a constant to every row of a table cannot change which row is smallest.

That is the precise sense in which ADR 0010's omission was and was not serious. For
**ranking** service levels at a fixed lead time it was harmless, which is why nothing
looked wrong. For **quoting a cost**, or for comparing a lead time against no lead time, it
understated the answer by a factor of eight. A number that is only valid as an ordering
should never have been printed in a column headed `total`.

The old arithmetic is recoverable and provably intact: 240,884 + 30,732 = 271,616, which is
exactly the total ADR 0010 published for that row.

## Cost

**Every published `--lead-time` figure in this repository is superseded**, including the
table in ADR 0010, which now reads as the transit-free special case rather than as the
answer. The instant-delivery numbers — which is every other number here — are untouched,
because with `lead_time_days=0` nothing is ever in transit and the new term is
multiplied by zero.

**The default is an upper bound, and a business with a real financing rate should pass
it.** Reporting the upper bound as though it were the cost would overstate the case
against long lead times, which is the mirror image of the error being fixed.

**Two rates now describe one warehouse**, and nothing enforces that the transit rate is
below the on-hand rate. It usually should be, and a caller who inverts them will get an
answer to a question about a very strange business rather than an error.

**The comparison a lead time invites is still not fully priced.** Cutting the lead time
from seven days to one would free roughly 38,000 units of working capital, and this model
now charges for holding that capital but still knows nothing about what shortening the lead
time would *cost* — expedited freight, a closer supplier, smaller and more frequent
deliveries. The pipeline is priced; the decision to shorten it is not.
