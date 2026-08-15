# 0008 — The simulator has no shipping lag, because the level already carries it

## Situation

An order-up-to system has two places a lead time can live.

It can live in the **level**: `S` is built to cover demand across the protection interval
`L + R`, so that stock ordered now carries the store until the next order can arrive.
`inventory/policy.py::order_up_to_level` does exactly this — a forward sum of the quantile
forecast over `lead_time + review_period`.

It can also live in the **loop**: an order placed on day `t` arrives on day `t + L`, and
until then the shelf has whatever it had.

Both are standard. Doing both to the same order is not: the store would be sized to
survive the wait *and* made to wait, and the protection interval would be paid for twice.

## Decision

The lead time lives in the level. `simulate` raises on-hand stock to `S` at the start of
every day with no delivery delay, serves demand, and loses whatever it could not serve.
`lead_time_days` is therefore a parameter of `order_up_to_level` and of `frontier`, which
builds levels — and is **not** a parameter of `simulate`, which consumes them. The stub
signature had it on `simulate`; it was removed rather than left in and ignored, because an
argument that changes nothing is worse than an absent one.

The committed specification tests forced the question and settled it. From
`tests/test_simulate.py`:

```python
demand = pd.Series([50.0, 200.0, 50.0])
result = simulate(demand, pd.Series([50.0] * 3), initial_stock=50.0)
assert result.stockout_days == 1
```

Day one is served, day two is short by 150, and day three is served again. Under any
model with a delivery delay, day three is short too and the answer is 2. The test is
asserting that a shortfall does not queue — the point of lost-sales retail — and it can
only do that if the shelf is refilled between the two.

## Cost

**Holding cost is biased high, and the bias is structural.** A real `(R, S)` system lets
stock cycle down between deliveries; average on-hand across a cycle is roughly half the
cycle stock plus the safety stock. Refilling daily holds the full protection interval
every day. The absolute currency figures the simulator prints are therefore not a budget.

**What survives the bias is the ordering**, which is what the frontier is read for. Every
curve — seasonal-naive, GBM point, GBM quantile — is simulated under the identical rule,
so "which curve sits below and to the right" is unaffected. The chart is a comparison, and
was only ever a comparison.

**A pipeline is not modelled at all**, so nothing here can show a bullwhip, and
`order_quantity` — which counts stock in transit precisely to avoid one — is exercised by
its own tests rather than by the simulator. If the project later needs absolute costs
rather than a ranking, this is the first thing to replace, and the fix is a delivery queue
plus a primed pipeline at `t = 0`.
