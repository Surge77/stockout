# Model card — stockout

**Status: five forecasters, evaluated on synthetic data only.** Every number below comes
from a generator, not from retail, and is stated that way wherever it appears.

## What exists

Three baselines, all fitted per store on trading days only.

| Model | Rule | Parameters |
|---|---|---|
| `naive_last` | Most recent trading-day figure | none |
| `seasonal_naive` | Most recent same store, same weekday | `season_length = 7` |
| `moving_average` | Mean of the last `window` trading days | `window = 28` |

Two gradient-boosted models, fitted on the horizon-aware feature matrix.

| Model | Objective | Parameters |
|---|---|---|
| `gbm` | `tweedie`, variance power 1.2 | 800 rounds, lr 0.05, 63 leaves, min 100 per leaf |
| `gbm_quantile` | `quantile` at each of 6 alphas | one booster per quantile, same tree settings |

Tweedie because sales are non-negative with a point mass at zero. Quantiles rather than a
mean plus `z * sigma`, because retail errors are neither symmetric nor constant-variance —
[ADR 0007](docs/decisions/0007-quantiles-not-point-forecast-plus-z-score.md).

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

Quantile calibration is measured, not assumed, and it is the weakest result in the repo.
Out of sample on the newest fold, every level under-covers:

| Nominal | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| Covered (trading days) | 0.357 | 0.564 | 0.607 | 0.721 | 0.843 | 0.893 |

Quantile crossing affected 42.9% of rows and was sorted before use.

## Limitations

- **The quantile models are not calibrated out of sample.** A nominal 0.9 covers about
  0.72 of trading days on held-out data. Stocking to a stated service level therefore does
  not deliver it, and the cost-minimising level on the sample lands at 0.90 rather than at
  the 0.75 the cost pair derives. Reported rather than corrected, because a post-hoc
  calibration shift would hide the fact that it happened.
- **Pooled by default, structured not at all.** The baselines fit per store. The
  gradient-boosted models are a single global fit with `store` as a feature, so they pool
  incidentally rather than by design — no hierarchy, no per-store effects, no shrinkage
  toward a group mean. Stores with short histories are served badly either way.
- **A flat forecast across the horizon.** Each baseline predicts one number per store (or
  per store-weekday) for all 42 days. It cannot represent a trend or an approaching event.
- **Promotions are ignored** by all three baselines, despite `promo` being available and
  future-known. The gradient-boosted models do use them.
- **The simulator has no delivery pipeline.** It prices a repeated single-period
  newsvendor, so its absolute currency figures are an ordering of options and not a
  budget — [ADR 0008](docs/decisions/0008-the-simulator-has-no-shipping-lag.md).
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

After that, in order: fixing the quantile calibration, and giving the simulator a delivery
pipeline so that absolute costs mean something.
