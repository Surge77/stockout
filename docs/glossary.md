# Glossary

Terms that mean something specific here, or that are routinely used loosely elsewhere.

## Forecasting

**Forecast origin** — the moment the forecast is made. Everything before it is known;
everything after is not. Nearly every leakage bug is a feature that quietly reaches past it.

**Horizon** — how far ahead the forecast reaches, in days. Set by the business (supplier
lead time), not chosen by the modeller. The default here is 42.

**Rolling origin** — a backtest that walks the forecast origin forward through history,
refitting at each step. The only kind of split this package offers.

**Expanding vs sliding window** — expanding keeps every fold anchored at the first
observation; sliding keeps a fixed-width training window. Expanding is the default; sliding
is better when there is an early regime change worth forgetting.

**Gap** — days deliberately left unused between train and test. Non-zero only when the
label takes time to settle. Rossmann's does not.

**Seasonal naive** — predict the same weekday last week. No fitting, no parameters, and on
retail data a genuinely strong forecast. The baseline everything here is measured against.

## Metrics

**WMAPE** — weighted mean absolute percentage error, `sum|error| / sum|actual|`. One
division, at the end. Primary metric here.

**MASE** — mean absolute scaled error. In this repository, model MAE over seasonal-naive
MAE **on the same evaluation window**, so below 1 means the baseline was beaten and 1.000
means it *is* the baseline. Classical MASE scales by in-sample naive MAE instead;
[ADR 0005](decisions/0005-wmape-and-mase-not-mape.md) explains the departure.

**RMSPE** — root mean squared percentage error, ignoring zero actuals. Kaggle's official
Rossmann metric, kept for comparability.

**Pinball loss** — quantile loss. Asymmetric by construction: at τ = 0.9 an under-forecast
costs nine times an over-forecast.

**Coverage** — the share of actuals falling at or below a predicted quantile. The
calibration check: a well-behaved 0.9 quantile is exceeded 10% of the time. A 0.9 quantile
covering 99% is not conservative, it is wrong, and it costs money in carried stock.

## Inventory

**Newsvendor critical ratio** — `Cu / (Cu + Co)`, the optimal service level for a
single-period stocking decision. `Cu` is the cost of being one unit short, `Co` the cost of
one unit left over. **This is the quantile to forecast** — derived from the cost pair, not
tuned.

**Order-up-to level (S)** — the base-stock target. Each review period, order enough to
bring the inventory position up to S.

**Inventory position** — stock on hand **plus stock already on order**. Counting only
on-hand re-orders everything in transit every cycle, which is the textbook bullwhip.

**Protection interval** — `lead time + review period`. Stock ordered now must cover demand
until the *next* order can arrive, not merely until this one does. Covering only the lead
time understocks by a whole review period.

**Lead time (L)** — days between placing an order and receiving it.

**Review period (R)** — days between opportunities to place an order.

**Fill rate** — share of demand met from stock. What a customer experiences.

**Cycle service level** — share of replenishment cycles with no stockout at all. What
planners usually quote. Always the higher-sounding of the two, which is why both are
reported.

**Stockout** — demand arriving with no stock to meet it. Here it is **lost, not
backordered**: retail walk-outs do not queue.

## Leakage

**Leakage** — using information at training time that would not exist at prediction time.
Three flavours are guarded against separately:

- **Split leak** — training on data later than the test set. A shuffled split.
- **Feature leak** — a rolling window that includes its own target row, or a lag shorter
  than the horizon.
- **Column leak** — a column present in the training file and absent at forecast time.
  `customers` is the canonical example.

**Denylist** — `features.build.FEATURE_DENYLIST`, the columns a model may never see.

**Future-known covariate** — a variable whose future values are already decided and
recorded, such as a promotion calendar. Using it is *not* leakage, and confusing it with a
lagged observation is the most common error in this problem.
