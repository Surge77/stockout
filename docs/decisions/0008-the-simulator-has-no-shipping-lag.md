# 0008 — The simulator prices a repeated newsvendor, not an (R, S) system

> **Extended by [ADR 0010](0010-the-pipeline-is-opt-in-and-the-critical-ratio-does-not-survive-it.md).**
> `simulate` now takes a `lead_time_days`, and everything below describes what it does at
> the default of `0`. Nothing here is retracted: the trap this ADR documents is still a
> trap, and 0010 exists because the two halves of the fix had to ship together.

## Situation

An order-up-to system can put the lead time in either of two places.

It can live in the **level**: `S` covers demand across the protection interval `L + R`, so
stock ordered now carries the store until the next order can arrive.
`inventory/policy.py::order_up_to_level` builds exactly that — a forward sum of the
quantile forecast over `lead_time + review_period`.

It can live in the **loop**: an order placed on day `t` arrives on day `t + L`, and until
then the shelf has whatever it had.

Both are standard. Doing both to the same order is not, and the committed specification
tests rule the loop version out. From `tests/test_simulate.py`:

```python
demand = pd.Series([50.0, 200.0, 50.0])
result = simulate(demand, pd.Series([50.0] * 3), initial_stock=50.0)
assert result.stockout_days == 1
```

Day one is served, day two is short by 150, day three is served again. Under any model
with a delivery delay, day three is short too and the answer is 2. The test asserts that a
shortfall does not queue — the point of lost-sales retail — and it can only do that if the
shelf is refilled between the two. The stub's docstring claimed a lead time in the loop;
the stub's tests contradicted it. The tests are the executable half, so they win.

## Decision

`simulate` raises on-hand stock to `S` at the start of every day, serves demand, and loses
what it could not serve. The protection interval is therefore one day, which makes each
day a **single-period stocking decision** — precisely the problem the newsvendor critical
ratio solves, and precisely what this project is named around. `frontier` stocks to each
day's own demand quantile and prices the outcome. `lead_time_days` appears on neither.

`order_up_to_level` remains, tested and documented, as the `(R, S)` base-stock formula.
It is the bridge to a pipeline simulator, not a component of this one.

## Cost

**The multi-period system is not simulated, so nothing here can show a bullwhip.**
`order_quantity`, which counts stock in transit precisely to avoid one, is exercised by
its own tests rather than by the simulator.

**Feeding an `(R, S)` level into this loop is a trap, and it was walked into once.** The
first frontier this repo produced applied `order_up_to_level` with the default `L=7, R=7`
and refilled daily. On store 1 of the committed sample that is a level of 96,159 against a
mean daily demand of 8,132 — **11.8 days of cover held every single day**. Every service
level returned a fill rate of 1.0000 and zero stockouts, the total-cost column became
monotone in the quantile, and the chart had no trade-off left in it.
`test_the_frontier_stocks_to_the_forecast_itself_not_to_a_multi_day_cover` exists so that
it cannot happen quietly again.

**Absolute currency figures are still not a budget.** Real stores hold safety stock across
a lead time; this one does not. What the frontier supports is the ordering of curves, and
every curve is simulated under the identical rule.
