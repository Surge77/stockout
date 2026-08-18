# 0010 — The delivery pipeline is opt-in, and the critical ratio does not survive it

> **Superseded by [ADR 0014](0014-a-scikit-learn-comparison-not-an-inventory-system.md).**
> The code this records — the quantile models, the conformal calibration and the
> inventory simulator — was removed in `c7fadcb`. The record is kept because a
> decision log that is edited to match the present is not a log; nothing below
> describes code that ships today.

## Situation

ADR 0008 chose a repeated newsvendor: stock is raised to the order-up-to level every
morning, demand arrives, unmet demand walks out, and the delivery is instant. That choice
was forced by the committed specification tests and it was the right one to make at the
time, but it left the module docstring's own promise unkept — absolute currency figures
were an ordering of curves and not a budget, because real stores wait for stock and this
one did not.

The trap that came with it is documented and pinned by a test. `policy.order_up_to_level`
sizes a base-stock level across the protection interval `L + R`; feeding that level to a
loop with instant delivery holds eleven days of cover every day, saturates every service
level at a fill rate of 1.0000, and flattens the frontier into a straight line. Two halves
of a model, and using either one alone is wrong.

## Decision

`simulate` gains `lead_time_days`, defaulting to `0`.

At `0` the loop is byte-for-byte the behaviour ADR 0008 describes: the inventory position
equals on-hand, the order lands the same morning, the protection interval is the single
day, and every test written against the repeated newsvendor still passes unchanged. That
remains the default because it is the decision the newsvendor critical ratio actually
solves, and this project is named around that decision.

Above `0` the loop opens a pipeline. An order placed on day `t` arrives on day `t + L`;
until then it is in transit — counted in the inventory position, absent from the shelf,
and not charged holding cost. Ordering against the position rather than against on-hand is
what stops the same stock being ordered seven mornings running, which is the textbook
bullwhip and is now covered by a hand-checkable test rather than only by
`policy.order_quantity`'s own unit tests.

**The two halves move together or not at all.** `frontier` re-sizes automatically: with a
lead time it stops stocking to the day's own quantile and calls `order_up_to_level` across
`L + 1` days, because this loop reviews daily. A caller who reaches past `frontier` into
`simulate` with a one-day level and a three-day wait gets a shelf that is empty most of
the window, and there is a test named after exactly that.

`initial_stock` defaults to opening in steady state — nothing when delivery is instant,
and the first day's base-stock level when it is not. A pipeline that starts empty is
guaranteed to be short on every day before the first lorry arrives, and charging a policy
for the warehouse having been built yesterday measures the opening balance rather than the
policy.

`SimulationResult` gains `mean_on_order`. A pipeline you cannot see in the output is a
pipeline you cannot check.

## What it showed, which was not what was expected

Store 1 of the committed sample, newest fold, `Cu = 3`, `Co = 1`:

| lead time | cheapest quantile | total cost | mean on hand | mean on order |
|---|---|---|---|---|
| 0 days | 0.80 | 43,456 | 539 | 0 |
| 7 days | 0.50 | 271,616 | 5,735 | 44,819 |

Costs rise by a factor of six and the cost-minimising service level collapses to the
bottom of the grid. Neither is a bug.

Holding cost is charged per unit **per day**, and a seven-day protection interval means
carrying roughly eight days of cover instead of one — so the holding term grows by about
the same factor while the shortage term, charged once per lost sale, does not. The
consequence is worth stating plainly because it contradicts something this repository
has repeated since ADR 0007:

**`Cu / (Cu + Co)` is the optimal service level for a single-period decision, and it stops
being the optimal service level the moment the protection interval is longer than one
day.** In a multi-period base-stock system the same unit is held and charged for many days
before it is either sold or written off, so the effective overage cost is multiplied by
the length of the interval and the optimal ratio falls. The critical ratio is not wrong;
it is being asked a question it does not answer. `policy.critical_ratio` still derives
0.75, the `L = 0` frontier still lands near it, and the `L = 7` frontier does not — and
that gap is a finding rather than a defect.

## Cost

**Two systems are now simulated by one function**, and the reader has to know which one
they asked for. Mitigated by the `frontier` header naming it, and by the default being the
one ADR 0008 argued for.

**In-transit stock is free here.** Nothing is charged for goods on a lorry, so the model
prefers a long pipeline to a full shelf. Real working capital does not work that way, and
a business that finances its inventory would price it.

**The `L = 7` frontier saturates at the top.** The 0.95 and 0.99 levels both return a fill
rate of 1.0000 with zero short days, so the last two rows carry no information beyond
their price. That is the honest shape of the answer at this cost pair rather than the ADR
0008 failure returning — the lower half of the curve still slopes, and
`test_the_frontier_sizes_across_the_protection_interval_once_deliveries_take_time`
asserts that it does.

**The review period for sizing is fixed at one day**, because the loop reviews daily. A
true `(R, S)` system that orders weekly would order on cycle boundaries and hold more
between them; that is a third system and it is not built.
