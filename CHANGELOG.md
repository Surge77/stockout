# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- `models/gbm.py` — LightGBM point (tweedie) and quantile forecasters
- `inventory/` — order-up-to policy, cost simulation, efficient frontier
- Answers to the five questions in `docs/questions.md`

## [0.1.0] — 2026-08-14

First scaffold. The spine runs end to end; the learned models do not exist yet.

### Added

- **Data** — Kaggle downloader with an error message naming both required steps; a
  deterministic synthetic generator reproducing weekly seasonality, Sunday closures, a
  refurbishment gap and a level shift; canonical schema with the `state_holiday`
  int-versus-string coercion; validation for dtypes, unique `(store, date)`, the
  `open == 0 ⟹ sales == 0` invariant, and calendar gaps.
- **Leakage guards** — `add_lags` raises `LeakageError` on a lag shorter than the horizon;
  `add_rolling` shifts before rolling; `FEATURE_DENYLIST` excludes `customers`;
  `split/rolling.py` offers no shuffled splitter and `assert_no_leakage` checks every fold.
- **Split** — rolling-origin and expanding-window folds with `gap` and `min_train_days`,
  raising `BacktestError` rather than silently returning fewer folds than requested.
- **Models** — `naive_last`, `seasonal_naive`, `moving_average`, all fitted on trading
  days and zeroed on closures.
- **Evaluation** — WMAPE, MASE, RMSPE, pinball, coverage; the rolling-origin backtest loop
  with a fresh model per fold; a markdown report that names the loser outright.
- **CLI** — `stockout fetch | synth | describe | backtest`.
- **Docs** — five questions committed before any analysis, seven ADRs, architecture, data
  dictionary, glossary, model card.
- **Tests** — 110 passing, 10 skipped, 98% coverage. Unit tests cannot open a socket.

### Notes

- `mase` uses the same-window form, so `seasonal_naive` scores exactly 1.000 and the
  harness self-checks. See ADR 0005.
- `wmape` is unaffected by including closed days — they cancel in numerator and
  denominator — while `mae` and `rmse` are flattered by them. Both are asserted.
- Stubs (`models/gbm.py`, `inventory/`) carry their reasoning and their skipped tests, so
  the specification is on record before the implementation.

[Unreleased]: https://github.com/Surge77/stockout/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Surge77/stockout/releases/tag/v0.1.0
