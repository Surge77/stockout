# Decisions

Each records the situation, what was chosen, and **what it cost**. A decision with no
downside listed is usually one that was not thought about hard enough.

| # | Decision | Cost |
|---|---|---|
| [0001](0001-data-is-downloaded-not-committed.md) | Data is downloaded, never committed | A fresh clone cannot reproduce the real numbers without a Kaggle account |
| [0002](0002-gradient-boosting-not-deep-learning.md) | Gradient boosting, not a sequence model | No cross-series learning; each store's history stands alone |
| [0003](0003-rolling-origin-only-no-random-splits.md) | Only rolling-origin splitters exist | Backtests cost 5x a single split, and small datasets get awkward |
| [0004](0004-lint-but-do-not-format.md) | `ruff check`, never `ruff format` | Formatting arguments are possible; nobody arbitrates them |
| [0005](0005-wmape-and-mase-not-mape.md) | WMAPE and MASE; MAPE excluded | Not directly comparable to work that quotes MAPE |
| [0006](0006-analysis-in-notebooks.md) | Analysis in notebooks, plumbing in the package | Notebook code is untested and reviewed by eye |
| [0007](0007-quantiles-not-point-forecast-plus-z-score.md) | Forecast quantiles directly | Five models instead of one, and quantile crossing to handle |
| [0008](0008-the-simulator-has-no-shipping-lag.md) | The lead time lives in the level, not in the loop | Holding cost is biased high; the frontier is an ordering, not a budget |
