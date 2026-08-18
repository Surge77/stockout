# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- The **loan classification task**: `Loan_Regression_Classification.xlsx` carries both
  `Loan_Approved` (binary) and `Loan_Amount` (continuous, and zero exactly when the first
  is), which is a paired classification-then-regression problem this package's machinery
  can already answer and has not been pointed at
- Answers to the five questions on **real** data. They are answered on the generator now,
  and three of the five came out flat or negative for reasons that are properties of the
  generator rather than of retail. This needs a Kaggle account and an accepted set of
  competition rules rather than any code
- Feeding the searched hyperparameters back into the registry. `tune` prints what it
  found; `linear.py` still says `alpha=1.0`, so a reader has to run the search to learn
  that the literal is not the searched answer. ADR 0017
- A hierarchical or per-store model. Every scikit-learn model here is one global fit with
  `store` as a feature, which pools incidentally and shares nothing deliberately — and Q2
  cannot test whether that is the right call on four stores
- Something to replace the decision layer ADR 0014 removed. Accuracy is not what a
  replenishment decision is graded on, and nothing here now says what a forecast costs

## [0.6.0] — 2026-08-18

The web app four docstrings had been promising. A **user** module that forecasts and an
**admin** module that trains and manages accounts, both served from the artefact the CLI
already wrote.

### Added

- **`src/stockout_web/`** — FastAPI, Jinja2 templates, a vendored htmx, and no build step.
  A separate package from `stockout` on purpose: `docs/architecture.md` claims everything
  between `data/` and `persistence.py` is pure, and a web app is nothing but I/O. The app
  imports the library and the library has never heard of the app.
- **The user page answers both problems at once.** One store-day in; predicted sales from
  the regression model *and* Low/Medium/High from the classification model out, with that
  store's own cut points beside them. The two are different models and can disagree, and a
  page showing one while implying both would be the most misleading thing here.
- **The admin page** — artefact provenance and held-out scores, retrain and redeploy with
  any pair from the registry, the whole-registry comparison table, and accounts.
- **Auth**: session cookie, signed, httpOnly, 8 hours; bcrypt with a per-password salt;
  SQLite with no ORM and every query parameterised. The role is read from the database on
  every request rather than from the cookie, so deactivating an account takes effect
  immediately. A wrong password and an unknown email give one message at equal cost. The
  last active admin can be neither demoted nor deactivated. No built-in default account.
- **A `web` CI job** that boots the app with uvicorn and probes it over HTTP. The unit
  tests drive handlers in process and would all pass while the server failed to start.
- **`pip install -e ".[web]"`** — an extra, because nothing in `src/stockout/` imports
  FastAPI and somebody who wants the models should not be given a web server.

### Changed

- **Coverage sources both packages.** `stockout_web` is held at 100% rather than 90%: it
  is where the auth lives, and a missing branch there is a way in rather than a number.
- **The four docstrings that started this** — in `dataset.py`, `comparison.py`,
  `adapter.py` and `persistence.py` — now name `stockout_web` and describe what it does,
  including that it never accepts an uploaded model.
- **`.env.example`** set `STOCKOUT_HORIZON_DAYS=42` against a default of 7 and carried an
  inventory section for the layer ADR 0014 deleted. Rewritten, with the web settings added.

### Fixed

- **`users.normalise_email("@")` was accepted**, because the check was `"@" in address`.
  Now requires something either side, exactly one `@`, and no whitespace — a sanity check
  rather than RFC 5322 validation, and the docstring says which.
- **The test suite's network guard blocked the event loop's own self-pipe.** Python builds
  it with `socket.socketpair()`, which on Windows is a connected 127.0.0.1 pair, so a
  blanket refusal made `TestClient` unusable while blocking nothing a test could reach. It
  now allows loopback and refuses everything else, which is what the guard always meant.

### Notes

- **There is no model upload and no data upload, deliberately.** `joblib.load` executes
  what it reads. An admin picks a model by name from the registry and this process writes
  the file; accepting a `.joblib` over HTTP would be arbitrary code execution.
- **The app starts with no model and no accounts**, because that is the first-run state —
  an admin has to sign in before there is anything to serve, so a missing artefact is a
  message on the page rather than a crash at boot.
- Training and the comparison run in a worker thread. Seconds of CPU with no `await` in
  them would otherwise block every other request, including the health check.

## [0.5.0] — 2026-08-18

The scikit-learn curriculum, finished: the registry reaches the command line, the notebook
runs again, and every document describes the package that ships. Three of the five
committed questions lost, which is reported at the top of `docs/results.md` rather than
somewhere further down.

### Added

- **Six subcommands, and one widened.** `compare` runs the whole registry against one
  holdout and names a winner; `leakage` prints the four-arm decomposition with its optimism
  column; `tune` grid searches over time-ordered folds; `prepare` builds and caches the
  model frame; `train` fits both tasks on everything and writes one artifact; `predict`
  reads it back and answers one store-day. `backtest --model` now accepts every registered
  regressor as well as the three baselines. Twelve regressors and nine classifiers were
  previously reachable only from a library import — and `persistence.py` told a failed load
  to run `python -m stockout train`, which did not exist.
- **Every model command routes through `dataset.prepare`**, which is what that module's
  docstring already claimed. That needed `--stores`: the real data is two files, four
  features come from the join, and `prepare` refuses to proceed without it rather than
  silently producing a frame with no categorical branch.
- **`--gap-days` defaults to the horizon**, not to zero. A holdout whose training rows end
  the day before its test rows begin is scoring a one-day forecast however long the horizon
  claims to be.
- **ADRs 0014 to 0020.** Four of them (0015, 0017, 0018, 0019) were cited from `src/` and
  `pyproject.toml` in eight places and had never been written. Each carries the measurement
  its argument rests on rather than the argument alone.
- **`tests/test_notebook.py`**, which execs every code cell into one namespace, checks each
  question wrote its figure, and asserts the notebook's question headings match the
  committed ones. No kernel: an `.ipynb` is JSON, and neither `nbclient` nor `ipykernel` is
  installed.

### Changed

- **`notebooks/01_explore.ipynb` rebuilt.** It imported six modules deleted by the previous
  release and raised on its first cell. Q5 no longer needs the inventory simulator — it now
  asks whether the classifier beats binning the regression, which is the question
  `predict._label` already leans on.
- **`docs/questions.md` says plainly that Q5 was rewritten after the comparison table
  existed**, and that Q1 to Q4 were not. A file whose whole point is that the questions
  predate the charts has to disclose the one that does not.
- **README, MODEL_CARD, architecture, glossary and results rewritten** against what runs,
  with every number regenerated. The README's three headline commands all exited 2; the
  glossary defined nine inventory terms the package no longer contains; the results file's
  leakage table quoted numbers no invocation reproduces.
- **CI's smoke job asserted two commands that no longer exist.** It ran `frontier` and
  `calibration`, grepped their stderr for a LightGBM install hint, and could only fail.
  Replaced with assertions about the current surface, including one that fails if a
  registered model is not offered by `--model`.
- **ADRs 0007 to 0013 keep their text and gain a supersession banner** naming 0014. A
  decision log edited to match the present is not a log.

### Fixed

- **A fold in which nobody runs a continuing promotion crashed the vectoriser.**
  `TfidfVectorizer` learned its vocabulary from the training documents, so a slice with no
  month tokens raised `empty vocabulary; perhaps the documents only contain stop words`
  four frames inside a `ColumnTransformer`. That is what fitting on a single store looks
  like, which the notebook does in Q2. The vocabulary is now fixed to
  `store_features.MONTH_TOKENS` — twelve months are a closed set known before any data is
  read — and the branch emits the same columns in the same order every fold. Identical
  output on the full sample.
- **`SimpleImputer` silently dropped a column with no observed values**, warning from four
  frames down. A store that has never run the promotion has `promo2_since_week` and
  `promo2_since_year` entirely null, and two features vanished from its matrix.
  `keep_empty_features=True` keeps them at zero instead.

  Both are the same bug in two branches: a transformer whose output width depends on which
  rows happened to land in the fold. Both were found by writing the notebook rather than by
  a test, and both now have one.

### Notes

- **The model ladder is flat, and that is the headline.** Twelve regressors land within
  0.01 WMAPE of each other; at four decimal places `linear` and the winner are the same
  number, 0.0899 each; `bagging` spends 7.7 seconds to finish below both. On a generator
  whose promotion calendar and weekday pattern are deterministic, a linear model on the
  right features is the correct answer, and reporting otherwise would be inventing a result.
- **Three of the five questions lost.** Accuracy does not decay with horizon here
  (0.0849 → 0.0835 → 0.0875 across 7 to 42 days, which was the stated failure condition);
  the fitted model beats the baseline on every store including the quiet ones; and the
  promotion wake is a −1.8% dip against a +24.5% lift, which is a reversal on the letter
  and barely on the substance. Q3 and Q5 held.
- **Q4's first answer was wrong and the write-up keeps it.** The wake counter ran straight
  through the following promotion and reported +22% persistence on days 7 and 8 — entirely
  the next cycle's lift. The counter now resets on any promotion day.
- **The leakage decomposition still finds almost nothing.** The shuffled split is optimistic
  by 0.0718 against the honest arm's 0.0566, and the preprocessing leak differs in the
  fourth decimal. Only the smuggled-`customers` arm is dramatic, at 0.998 with a *negative*
  optimism — a protocol not lying about its own error at all, attached to a model that is
  undeployable the same afternoon.

## [0.4.0] — 2026-08-17

The three defects the last release wrote down, closed. Two of them produced numbers that
contradict what this repository previously argued, and both contradictions are in the
release notes rather than under them.

### Added

- **A charge on stock in transit** — `simulate(..., transit_holding_cost=...)`, defaulting
  to the on-hand rate because committed capital earns nothing on a lorry, with `0.0` for a
  supplier-owned pipeline. Reported as its own `transit_cost` column, never blended into
  holding. `--transit-holding-cost` on the command line. ADR 0011.
- **Conditional coverage** — `metrics.coverage_by_segment` and
  `report.conditional_coverage_to_markdown` score each store or month at each level with
  the row count the number rests on. `stockout calibration --by store|month`. ADR 0012.
- **Per-group calibration** — `ConformalQuantileForecaster(group_by=...)` learns a Mondrian
  offset per group above a floor *derived* from the grid rather than chosen:
  `conformity.min_rows_for` returns the count at which the strictest level stops
  saturating. `pooled_fallback_groups` and `unseen_groups` name every group that took the
  marginal offset instead. `--calibrate-by store`. ADR 0012.
- **A calibration that carries its theorem** — `ConformalQuantileForecaster(refit=False)`
  serves the probe itself, so the offsets describe the estimator that produced them. Made
  possible by `design.GbmDesign` separating *boosting rows* from *feature history*: the
  model trains on the inner window while still lagging across the calibration window, which
  is the alignment problem ADR 0009 thought made this impossible. `--no-refit`. ADR 0013.

### Changed

- `stockout.__version__` is read from the installed distribution. It had said `0.1.0` since
  the first release while `pyproject.toml` moved to `0.3.0` — hand-maintained in two places,
  wrong in one, and silent because nothing imports a version to check it.
- Six files split by responsibility to stay inside the 300-line limit, no behaviour
  changed: `inventory/frontier.py`, `models/conformity.py`, `models/design.py`,
  `models/protocols.py`, `commands.py`, and `test_cli.py` into three files.

### Notes

- **The transit charge multiplies lead-time costs eightfold and moves no ranking.** At a
  seven-day lead time the cheapest level's total goes 271,616 → 2,153,995 and the cheapest
  level stays 0.50. In steady state the pipeline holds throughput × lead time and
  throughput is demand, which the service level does not change, so `transit_cost` varies
  by 2.6% across the grid while `holding_cost` more than doubles. Omitting it was harmless
  for ranking service levels and wrong by a factor of eight for quoting a cost.
- **The marginal coverage number this repository has been publishing understated the worst
  store by about two thirds.** Raw: worst marginal gap −0.193, worst store −0.314. After
  calibration: −0.121 marginal against −0.207 for store 1. A pooled offset also has to be
  wrong in two directions at once — store 2 now over-covers at three levels while store 1
  under-covers at all six.
- **Marginal calibration improved the conditional picture anyway**, narrowing the per-store
  spread at the 0.90 from 0.229 to 0.115. Not luck: ADR 0009 divides each residual by the
  model's own prediction, so the correction is relative and already scales with store level.
  It was justified on heteroscedasticity grounds and bought conditional validity too.
- **ADR 0009 predicted that serving the probe would be "a worse trade". It is not.**
  Coverage improves at every level from the 0.80 up (0.90: −0.121 → −0.086; 0.99: −0.061 →
  −0.026) and degrades at the median. The cost lands on the point forecast and lands
  consistently — WMAPE 0.0723 → 0.0758, MAE 674.4 → 707.2, median pinball 337.2 → 353.6,
  all three about 5%, which is what dropping 42 of ~600 training days buys.
- **Those coverage differences are about 1.4 sigma on 140 test rows**, so `refit=True`
  remains the default. Moving a default on that evidence is the error ADR 0009 refused when
  it declined to pick a row threshold from four draws.
- **On this data the per-group correction declines to act.** Four stores hold 35 trading
  rows each against a derived floor of 199, so every group falls back to pooled and the
  command says so. That is the honest result; Rossmann is worse per store, not better.

## [0.3.0] — 2026-08-16

The two defects the last release wrote down, fixed — one of them only halfway, and the
number that says so is in the release notes rather than under it.

### Added

- **Calibration** — `ConformalQuantileForecaster` (`--model gbm_conformal`), conformal
  arithmetic with a scaled conformity score, a probe model fitted on the inner window and
  a deployed model fitted on all of it. That refit gives up the split-conformal theorem in
  exchange for correct lag alignment, so the coverage claim is measured rather than
  proven, and ADR 0009 says so in its own words. `offsets`, `calibration_rows` and
  `saturated_quantiles` are public, because a correction nobody can inspect is a
  correction nobody can defend. ADR 0009.
- **`stockout calibration`** — nominal against empirical coverage for the raw and the
  calibrated model side by side, plus pinball loss per level and a warning naming any
  level whose offset rests on a single observation.
- **Delivery pipeline** — `simulate(..., lead_time_days=...)`. Orders are placed against
  the inventory position, arrive `L` days later, and are visible in the new
  `mean_on_order`. `frontier` re-sizes across the protection interval and opens in steady
  state when a lead time is set, because the two halves have to move together. ADR 0010.
- **`metrics.coverage_table`** and **`report.calibration_to_markdown`**, promoted out of
  the notebook cell that had been computing coverage by hand.

### Changed

- `notebooks/01_explore.ipynb` executes end to end. Q3 and Q4 were hints and are now
  cells; Q5 prices three policies rather than one; the calibration cell calls the package.
- ADR 0008 carries a pointer to ADR 0010. Nothing in it is retracted — the trap it
  documents is still a trap, which is why the pipeline and the re-sizing shipped together.

### Notes

- **Calibration moves the cost-minimising service level from 0.90 to 0.80** and cuts the
  fold's total cost from 44,271 to 43,456. ADR 0007 predicted the optimum should sit at
  the 0.75 the cost pair derives; the raw model put it at 0.90; fixing the coverage moved
  it towards the prediction. Pinball loss improves at every level too.
- **It is a half-fix.** A nominal 0.9 covered 0.721 and now covers 0.779. The rest is
  distribution shift between the calibration window and the test window that follows it.
- **The number that disagrees.** On a two-store draw with 70 calibration rows, calibration
  is worse than no calibration — worst gap 0.107 → 0.171, pinball 188.9 → 195.6. It helps
  at 105, 140 and 175 rows. No threshold is enforced; the row count is printed instead.
- **A seven-day lead time collapses the cheapest level to 0.50** and multiplies cost
  sixfold. Holding is charged every day of the protection interval and shortage once, so
  `Cu / (Cu + Co)` stops being the right target the moment that interval exceeds a day.
- Q4's original design divides by an empty set on any calendar that promotes every other
  week, and its replacement put the same fault in the denominator — a weekday reference
  built from every non-promotion day contains the wake it is meant to measure, and returns
  a lift of 1.000 whatever is true. The third version excludes the wake from the reference
  and prints `PROFILE NOT ESTIMABLE` on this sample rather than a reassuring flat line.
  Q4 needs promotion sparsity, not just promotion data.
- 328 tests, none skipped, 98% coverage.

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

[Unreleased]: https://github.com/Surge77/stockout/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Surge77/stockout/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Surge77/stockout/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Surge77/stockout/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Surge77/stockout/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Surge77/stockout/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Surge77/stockout/releases/tag/v0.1.0
