# Model card — stockout

**Status: baselines only.** No learned model has been trained. This card describes what
exists today and states plainly what does not, so that nothing here can be mistaken for a
result. It is updated when the gradient-boosted models land.

## What exists

Three baselines, all fitted per store on trading days only.

| Model | Rule | Parameters |
|---|---|---|
| `naive_last` | Most recent trading-day figure | none |
| `seasonal_naive` | Most recent same store, same weekday | `season_length = 7` |
| `moving_average` | Mean of the last `window` trading days | `window = 28` |

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

On the synthetic sample: `seasonal_naive` WMAPE 0.1489 (MASE 1.000 by definition),
`moving_average` 1.080, `naive_last` 1.144.

**These are synthetic-data figures and are not evidence about retail forecasting.**

## Limitations

- **No learned model.** Any claim about gradient boosting in this repository is a plan,
  not a measurement.
- **Per-store, no pooling.** Stores with short histories are served badly and nothing
  shares strength across series.
- **A flat forecast across the horizon.** Each baseline predicts one number per store (or
  per store-weekday) for all 42 days. It cannot represent a trend or an approaching event.
- **Promotions are ignored** by all three baselines, despite `promo` being available and
  future-known.
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

Implementing `models/gbm.py`, then reporting the backtest honestly — including the
outcome where the gradient-boosted model **loses** to seasonal-naive, which is a real
possibility on data with this much weekly structure and would be reported as loudly as a
win.
