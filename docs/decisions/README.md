# Decisions

Each records the situation, what was chosen, and **what it cost**. A decision with no
downside listed is usually one that was not thought about hard enough.

## Current

| # | Decision | Cost |
|---|---|---|
| [0001](0001-data-is-downloaded-not-committed.md) | Data is downloaded, never committed | A fresh clone cannot reproduce the real numbers without a Kaggle account |
| [0002](0002-gradient-boosting-not-deep-learning.md) | Gradient boosting, not a sequence model | No cross-series learning; each store's history stands alone — *the library named in it is amended by 0018* |
| [0003](0003-rolling-origin-only-no-random-splits.md) | Only rolling-origin splitters exist | Backtests cost 5x a single split, and small datasets get awkward — *`split/strategies.py` is the one exemption, and 0020 is why* |
| [0004](0004-lint-but-do-not-format.md) | `ruff check`, never `ruff format` | Formatting arguments are possible; nobody arbitrates them |
| [0005](0005-wmape-and-mase-not-mape.md) | WMAPE and MASE; MAPE excluded | Not directly comparable to work that quotes MAPE |
| [0006](0006-analysis-in-notebooks.md) | Analysis in notebooks, plumbing in the package | Notebook code is untested and reviewed by eye — *`tests/test_notebook.py` now runs every cell, which is not the same as reviewing them* |
| [0014](0014-a-scikit-learn-comparison-not-an-inventory-system.md) | A scikit-learn comparison, not an inventory system | The project's strongest claim is gone; seven ADRs and every published cost figure describe deleted code |
| [0015](0015-a-subsample-is-reported-never-silent.md) | A model's row cap travels with its score | The capped models are judged unfairly and the table says so rather than fixing it; the cap never binds on the committed sample |
| [0016](0016-demand-classes-are-per-store-terciles-fitted-on-the-training-window.md) | Demand classes are per-store terciles, fitted on the training window | Every accuracy reads as mediocre against a 33% floor; a store with no training rows gets the pooled cut points this ADR argues against |
| [0017](0017-every-hyperparameter-is-searched-not-defaulted.md) | Every hyperparameter is searched over time-ordered folds | On the committed sample the search buys provenance and not accuracy — it picked a value 0.002 R² *worse* than the default |
| [0018](0018-histgradientboosting-replaces-lightgbm.md) | `HistGradientBoosting` replaces LightGBM | Every historical gradient-boosted number came from a library this project can no longer install |
| [0019](0019-store-is-an-ordinal-id-not-1115-one-hot-columns.md) | `store` is an ordinal id, not 1115 one-hot columns | Linear models get a column whose magnitude is meaningless; a tree can still carve out one shop |
| [0020](0020-the-leakage-experiment-holds-the-future-fixed.md) | The leakage arms are scored on one fixed future | Three of four arms return a null result on this data; only the arm that cheats outright is dramatic |

## Superseded

Kept rather than deleted. A decision log edited to match the present is not a log — and a
commit message from 2026 that links to one of these should land somewhere that explains
what happened rather than at a 404.

| # | Decision | Status |
|---|---|---|
| [0007](0007-quantiles-not-point-forecast-plus-z-score.md) | Forecast quantiles directly | Superseded by 0014 |
| [0008](0008-the-simulator-has-no-shipping-lag.md) | The simulator prices a repeated newsvendor | Superseded by 0014 |
| [0009](0009-conformal-calibration-not-a-recalibrated-loss.md) | Calibrate the quantiles conformally | Superseded by 0014 |
| [0010](0010-the-pipeline-is-opt-in-and-the-critical-ratio-does-not-survive-it.md) | The delivery pipeline is opt-in | Superseded by 0014 |
| [0011](0011-stock-in-transit-is-not-free.md) | Stock in transit is charged at the shelf rate | Superseded by 0014 |
| [0012](0012-coverage-is-measured-per-group-and-corrected-per-group-only-when-the-rows-allow.md) | Coverage measured per group, corrected only above a derived floor | Superseded by 0014 |
| [0013](0013-the-refit-is-optional-and-dropping-it-buys-back-one-of-two-premises.md) | The refit is optional | Superseded by 0014 |
