# Model card — stockout

**Status: six forecasters, evaluated on synthetic data only.** Every number below comes
from a generator, not from retail, and is stated that way wherever it appears.

## What exists

Three baselines, all fitted per store on trading days only.

| Model | Rule | Parameters |
|---|---|---|
| `naive_last` | Most recent trading-day figure | none |
| `seasonal_naive` | Most recent same store, same weekday | `season_length = 7` |
| `moving_average` | Mean of the last `window` trading days | `window = 28` |

Three gradient-boosted models, fitted on the horizon-aware feature matrix.

| Model | Objective | Parameters |
|---|---|---|
| `gbm` | `tweedie`, variance power 1.2 | 800 rounds, lr 0.05, 63 leaves, min 100 per leaf |
| `gbm_quantile` | `quantile` at each of 6 alphas | one booster per quantile, same tree settings |
| `gbm_conformal` | `gbm_quantile` plus a per-level offset | offsets from a 42-day held-out tail, scaled by the median prediction |

Tweedie because sales are non-negative with a point mass at zero. Quantiles rather than a
mean plus `z * sigma`, because retail errors are neither symmetric nor constant-variance —
[ADR 0007](docs/decisions/0007-quantiles-not-point-forecast-plus-z-score.md). Conformal
offsets on top of those quantiles because, measured, the stated levels did not hold —
[ADR 0009](docs/decisions/0009-conformal-calibration-not-a-recalibrated-loss.md).

Features are calendar covariates that are knowable in advance, plus lags and rolling
statistics that are all at least one horizon old. `customers` is excluded by a denylist
enforced by a test.

Every one predicts zero when the trading calendar says the store is shut. That calendar is
future-known — it comes from a planning system, not an observation — so using it is not
leakage.

## Intended use

Establishing the floor that a learned model has to clear, and exercising the pipeline end
to end. **Not** for making real replenishment decisions.

## Training data

The committed sample is **synthetic** (`stockout.data.synth`, 4 stores × 730 days,
seed 7). It deliberately reproduces four structures — weekly seasonality, Sunday closures,
a refurbishment gap where rows vanish, and a permanent level shift — and nothing else.

Real runs use Rossmann Store Sales (1115 German drugstores, 2013-01-01 to 2015-07-31),
downloaded per machine and never redistributed.

## Evaluation

Rolling-origin backtesting, 5 folds, 42-day horizon, 365-day minimum expanding training
window. Closed days excluded from scoring. Metrics: WMAPE (primary), MASE against
seasonal-naive, RMSPE (Kaggle's metric). MAPE is deliberately absent —
[ADR 0005](docs/decisions/0005-wmape-and-mase-not-mape.md).

On the synthetic sample:

| Model | WMAPE | MASE | RMSPE |
|---|---|---|---|
| `gbm` | 0.0793 | **0.533** | 0.1018 |
| `gbm_quantile` | 0.0811 | 0.545 | 0.1053 |
| `seasonal_naive` | 0.1489 | 1.000 | 0.1931 |
| `moving_average` | 0.1606 | 1.080 | 0.2195 |
| `naive_last` | 0.1695 | 1.144 | 0.2177 |

**These are synthetic-data figures and are not evidence about retail forecasting.** The
generator's promotion calendar and weekday pattern are deterministic, so a model with
calendar features has an advantage here that it would have to re-earn on Rossmann.

Quantile calibration is measured, not assumed, and it remains the weakest result in the
repo. Out of sample on the newest fold, every level under-covers, and calibration closes
about half of every gap:

| Nominal | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| `gbm_quantile` covers | 0.357 | 0.564 | 0.607 | 0.721 | 0.843 | 0.893 |
| `gbm_conformal` covers | 0.414 | 0.650 | 0.693 | 0.779 | 0.886 | 0.929 |

Pinball loss improves at every level as well (350.8 → 337.2 at the median, 74.4 → 42.3 at
the 0.99), so the coverage is not bought by degrading the forecast. Reproduce with
`python -m stockout calibration`.

Quantile crossing affected 42.9% of rows and was sorted before use.

## Limitations

- **Even calibrated, the quantiles under-cover.** A nominal 0.9 covers 0.721 raw and 0.779
  after conformal correction. Calibration moved the cost-minimising level from 0.90 to
  0.80 — towards the 0.75 the cost pair derives, which is what ADR 0007 predicted — but did
  not close the distance. The remainder is distribution shift between the calibration
  window and the test window that follows it.
- **Calibration is conditional on having enough rows, and can hurt without them.** On a
  two-store draw with 70 calibration rows, both the worst coverage gap and the pinball loss
  got worse. `calibration_rows` and `saturated_quantiles` are public and printed; no
  threshold is enforced, because one picked from four draws would be a guess.
- **Coverage is corrected marginally by default, and the marginal figure understates the
  worst store by about two thirds.** Measured per store, the raw model's worst gap is 0.314
  against a marginal 0.193, and the calibrated model's is 0.207 against a marginal 0.121.
  Store 2 over-covers at three levels while store 1 under-covers at all six, which is what a
  single additive offset must do when asked to move a quiet shop and a busy one in opposite
  directions. `stockout calibration --by store|month` prints the grid; `--calibrate-by
  store` corrects per store where the rows allow, which on four synthetic stores is nowhere
  — [ADR 0012](docs/decisions/0012-coverage-is-measured-per-group-and-corrected-per-group-only-when-the-rows-allow.md).
- **Season-shaped miscalibration is measurable and not correctable.** The calibration window
  is the 42 days before the test window and contains none of the months being predicted, so
  there is no December residual with which to correct December.
- **The coverage claim is measured, not proven, and `--no-refit` removes one of the two
  reasons why.** Split conformal proves marginal coverage for the estimator whose residuals
  were used; by default the deployed model is refitted on more data, so the theorem does not
  transfer. `refit=False` serves the estimator the residuals describe and recovers that
  half, at a cost of about 5% on WMAPE, MAE and median pinball alike. What remains is
  exchangeability between the calibration and test rows, which a test window following a
  calibration window does not have — so no figure here is a guarantee
  ([ADR 0009](docs/decisions/0009-conformal-calibration-not-a-recalibrated-loss.md),
  [ADR 0013](docs/decisions/0013-the-refit-is-optional-and-dropping-it-buys-back-one-of-two-premises.md)).
- **Pooled by default, structured not at all.** The baselines fit per store. The
  gradient-boosted models are a single global fit with `store` as a feature, so they pool
  incidentally rather than by design — no hierarchy, no per-store effects, no shrinkage
  toward a group mean. Stores with short histories are served badly either way.
- **A flat forecast across the horizon.** Each baseline predicts one number per store (or
  per store-weekday) for all 42 days. It cannot represent a trend or an approaching event.
- **Promotions are ignored** by all three baselines, despite `promo` being available and
  future-known. The gradient-boosted models do use them.
- **The delivery pipeline is opt-in.** By default the simulator prices a repeated
  single-period newsvendor
  ([ADR 0008](docs/decisions/0008-the-simulator-has-no-shipping-lag.md)); `--lead-time`
  opens a real pipeline, and under one the cost-minimising service level is no longer the
  critical ratio at all —
  [ADR 0010](docs/decisions/0010-the-pipeline-is-opt-in-and-the-critical-ratio-does-not-survive-it.md).
- **Stock in transit is charged at the shelf rate, which is a bound rather than an
  estimate.** On-hand stock also buys warehouse space, insurance and shrinkage and a lorry
  buys none of them, so the true transit rate is lower; any specific fraction would be
  invented, so the conservative bound is the default and `--transit-holding-cost` accepts the
  real one. The charge takes a seven-day lead time's cheapest total from 271,616 to
  2,153,995 and moves no ranking, because the pipeline is throughput × lead time and
  throughput does not depend on the service level
  ([ADR 0011](docs/decisions/0011-stock-in-transit-is-not-free.md)).
- **Shortening the lead time is not priced at all.** Holding the pipeline costs something
  now; expedited freight or a closer supplier still costs nothing, so the model can say a
  long pipeline is expensive and not what to do about it.
- **Synthetic evaluation only**, so far.

## Ethical and practical considerations

Store-level revenue aggregates; no personal data, and no individual is identifiable. The
`customers` column is a daily count, not a record of people, and is excluded from the
feature set anyway.

The realistic harm from a deployed version of this is commercial: an under-forecast
empties shelves, an over-forecast ties up cash and, for perishables, creates waste. That
asymmetry is exactly what the newsvendor cost pair encodes, and it is why the target
service level is derived from costs rather than defaulted to a round number.

## What would change this card

Running any of it on the real Rossmann file. Every figure above is a statement about
`stockout.data.synth`, and the five questions in [docs/questions.md](docs/questions.md)
stay unanswered until they can be asked of real data.

Every follow-up this card has ever listed has now been done — the quantiles are calibrated,
the simulator has a pipeline, the pipeline is charged for, coverage is measured per group,
and the refit that cost the theorem is optional. Each turned out to be a partial fix with its
remainder written down, and two of them produced numbers contradicting what an earlier
version of this card asserted: the marginal coverage figure was understating the worst store
by two thirds, and serving the probe was supposed to be a worse trade and is not.

What is left is not a modelling change. It is `python -m stockout fetch`, one line at the top
of the notebook, and five answers read off cells that already run.
