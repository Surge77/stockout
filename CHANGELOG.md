# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Answers to the five questions in `docs/questions.md`, which need the real Rossmann file
- Calibration of the quantile models, whose out-of-sample coverage is measurably short
- A delivery pipeline in `inventory/simulate.py`, so that absolute costs mean something

## [0.2.0] — 2026-08-15

The unbuilt half. Every `NotImplementedError` is gone and every skipped test is green.

### Added

- **Models** — `GbmForecaster` (tweedie) and `GbmQuantileForecaster` (one booster per
  quantile, crossing sorted and its rate reported). `fit` retains the training frame,
  because a lag of at least one horizon means a test window's features resolve into
  history and cannot be built from the future frame alone.
- **Inventory** — `order_up_to_level`, `order_quantity`, `simulate` and `frontier`. The
  simulator walks stock forward day by day, loses unmet demand rather than backordering
  it, and prices holding against shortage.
- **CLI** — `stockout frontier`, and `--model gbm | gbm_quantile` on `backtest`, wired
  through a name-to-factory registry in `models/__init__.py`.
- **Report** — `frontier_to_markdown`, which names the cheapest service level rather than
  leaving a reader to scan for it.
- **ADR 0008** — the simulator prices a repeated newsvendor, not an `(R, S)` system.
- **Smoke CI** — a bare install must fail `frontier` readably, which is the end-to-end
  proof that LightGBM is imported lazily.

### Changed

- `dev` now self-references the `gbm` extra: the gradient-boosted models are real code,
  and code CI cannot execute is code CI cannot measure.
- `DEFAULT_QUANTILES` gains 0.75 — the ratio the default cost pair derives, which the
  grid previously could not price.
- `simulate` no longer takes `lead_time_days`, and `frontier` no longer applies a
  protection interval. See the note below.

### Notes

- On the synthetic sample `gbm` scores **MASE 0.533**, beating seasonal-naive by 46.7%.
  Discount it: the generator's promotion calendar and weekday pattern are deterministic.
- **The quantile models under-cover out of sample** — a nominal 0.9 covers 0.721 of
  trading days. The cost-minimising level therefore lands at 0.90 rather than the 0.75 the
  cost pair derives. Recorded in `docs/results.md` rather than calibrated away.
- The stub's docstring described a delivery lead time inside `simulate`; the stub's own
  test required that a shortfall not carry into the next day. Those cannot both hold. The
  test won, and ADR 0008 records what that costs.
- The first frontier built from an `(R, S)` level held 11.8 days of cover every day and
  returned a fill rate of 1.0000 at every service level. A regression test now pins it.
- 283 tests, none skipped, 99% coverage.

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
- **Tests** — 232 passing, 10 skipped, 98% coverage. Unit tests cannot open a socket, and
  an AST check asserts that no randomised split executes anywhere in the package.

### Notes

- `mase` uses the same-window form, so `seasonal_naive` scores exactly 1.000 and the
  harness self-checks. See ADR 0005.
- `wmape` is unaffected by including closed days — they cancel in numerator and
  denominator — while `mae` and `rmse` are flattered by them. Both are asserted.
- Stubs (`models/gbm.py`, `inventory/`) carry their reasoning and their skipped tests, so
  the specification is on record before the implementation.

[Unreleased]: https://github.com/Surge77/stockout/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Surge77/stockout/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Surge77/stockout/releases/tag/v0.1.0
