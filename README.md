# stockout

[![CI](https://github.com/Surge77/stockout/actions/workflows/ci.yml/badge.svg)](https://github.com/Surge77/stockout/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Coverage 98%](https://img.shields.io/badge/coverage-98%25-brightgreen)](#testing)
[![Checked with pyright](https://img.shields.io/badge/types-pyright-blue)](https://github.com/microsoft/pyright)

Forecasts retail demand to drive **replenishment**, and is scored on the decision — stock
carried and sales lost — rather than on a percentage error. Validated by rolling-origin
backtesting, with no random splits anywhere in the package.

Named after the failure it exists to prevent.

```bash
pip install -e ".[dev]"
python -m stockout describe                            # what's in the sample
python -m stockout backtest --model gbm                # beats the baseline by 47%
python -m stockout calibration                         # does a stated 0.9 cover 90%? no
python -m stockout frontier --model gbm_conformal      # what that forecast costs to stock
```

## Why this project exists

Most beginner forecasting projects are ordinary tabular regression that happens to have a
date column. Shuffle the rows, fit a model, report R². The score is excellent and it is
not a forecast, because the model was shown next month while predicting last month.

Time series breaks the assumption every other tabular project rests on:
`train_test_split(shuffle=True)` stops being a style preference and becomes a **bug**. That
one difference is why this project is worth building and nine near-identical
price-prediction projects are not.

Two things follow from it, and they are the whole content:

**Leakage is caught by the test suite, not by discipline.** Three separate guards, each
for a different class of mistake. The most useful is a property test: perturb the newest
target value and assert that *no feature anywhere changes*. Any leak, however written,
breaks it — and its partner test asserts that perturbing an *old* value **does** change
later features, so the first cannot pass by being vacuous.

**A forecast is judged by the decision it drives.** Low WMAPE and low cost are different
orderings, and a forecast can win the first while losing the second. The stocking quantile
here is not a hyperparameter — it is the newsvendor critical ratio `Cu / (Cu + Co)`,
derived from what a lost sale costs versus what carrying stock costs. Change the business,
the target moves; rerun a grid search, it does not.

## What's in here, and what isn't

**In the package** — acquisition, schema validation, cleaning, splitting, feature
construction, metrics, the backtest loop, three baselines, a LightGBM point model, a
quantile model, the conformal calibration that makes its stated levels mean something, and
the inventory simulator that prices what stocking to each level costs. Everything with one
correct answer, tested and type-checked.

**In the notebook** — the analysis. The five questions are committed in
[docs/questions.md](docs/questions.md) *before* any chart exists, so findings cannot be
retrofitted to whatever turned up. See [ADR 0006](docs/decisions/0006-analysis-in-notebooks.md).

**Not answered yet** — those five questions. Every cell that answers them runs end to end;
what is missing is data worth answering them from, because every number in this repository
was produced by a generator. That is one `python -m stockout fetch` and one changed line at
the top of the notebook away. See [Known limits](#known-limits).

## Current state

Everything below comes from `python -m stockout backtest` on the committed synthetic
sample — 4 stores, 730 days, 5 rolling-origin folds, 42-day horizon.

| Model | Mean WMAPE | MASE | |
|---|---|---|---|
| `gbm` | 0.0793 | **0.533** | beats the baseline by 46.7% |
| `gbm_quantile` | 0.0811 | 0.545 | the median of the quantile fit |
| `seasonal_naive` | 0.1489 | 1.000 | the baseline, by definition |
| `moving_average` | 0.1606 | 1.080 | 8.0% worse |
| `naive_last` | 0.1695 | 1.144 | 14.4% worse |

The gap between `seasonal_naive` and `naive_last` is the value of knowing what day of the
week it is. The gap at the top is worth less than it looks: this generator's promotion
calendar and weekday pattern are deterministic, so a model with calendar features is being
handed most of the answer. On Rossmann it would have to earn it again.

And accuracy is not the deliverable. `python -m stockout frontier` prices what stocking to
each quantile actually costs, on the newest fold, for one store:

| quantile | fill rate | stockout days | holding | shortage | **total** |
|---|---|---|---|---|---|
| 0.50 | 0.945 | 22 | 8,794 | 56,291 | 65,086 |
| 0.75 | 0.963 | 18 | 15,238 | 37,626 | 52,864 |
| 0.80 | 0.971 | 18 | 17,720 | 30,274 | 47,994 |
| 0.90 | 0.980 | 10 | 23,907 | 20,364 | **44,271** |
| 0.95 | 0.984 | 8 | 30,102 | 16,318 | 46,420 |
| 0.99 | 0.990 | 6 | 38,926 | 10,544 | 49,470 |

Cost is U-shaped in the service level, so there is a cheapest place to stand and it is not
"as accurate as possible". That curve is the project.

**It was also not where the theory said it should be.** The cost pair (`Cu` 3, `Co` 1)
derives a critical ratio of 0.75, and 0.75 was not the cheapest row — 0.90 was. The reason
is measurable rather than mysterious: the quantile models under-cover out of sample, so a
nominal 0.9 delivered about 0.72 and you had to over-ask to land on the service level you
wanted.

`python -m stockout calibration` measures it, and `--model gbm_conformal` corrects it with
split-conformal offsets learned on a held-out tail of the training window:

| nominal | 0.50 | 0.75 | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|
| raw covers | 0.357 | 0.564 | 0.607 | 0.721 | 0.843 | 0.893 |
| calibrated covers | 0.414 | 0.650 | 0.693 | 0.779 | 0.886 | 0.929 |

That moves the cheapest service level from **0.90 to 0.80** and the fold's total cost from
44,271 to 43,456 — the theory's prediction was right and the model was wrong, which is the
better way round. It is a half-fix: a nominal 0.9 still delivers 0.779, and on a two-store
draw with only 70 calibration rows the correction is *worse* than no correction at all.
Both numbers are in [docs/results.md](docs/results.md) and
[ADR 0009](docs/decisions/0009-conformal-calibration-not-a-recalibrated-loss.md), reported
rather than smoothed over.

**Add a delivery lag and the target moves again.** `--lead-time 7` opens a real order
pipeline and sizes the base-stock level across the protection interval; costs rise sixfold
and the cheapest level drops to 0.50, because holding is charged on every day of that
interval and a lost sale only once. `Cu / (Cu + Co)` is the answer to a single-period
question and stops being the answer when the interval is longer than a day
([ADR 0010](docs/decisions/0010-the-pipeline-is-opt-in-and-the-critical-ratio-does-not-survive-it.md)).

## The four traps this data sets

1. **A closed store sells zero.** About a seventh of Rossmann rows are closures. Every
   model predicts them perfectly, which flatters `mae` and `rmse` — they are per-row
   averages. `wmape` is immune, because a closed day cancels in both numerator and
   denominator. Both behaviours are asserted, because assuming either one is easy.
2. **`customers` is bait.** It predicts `sales` at about r = 0.9, it is in the training
   file, and nobody knows it six weeks ahead. It is on a denylist enforced by a test.
3. **A lag shorter than the horizon is unobservable.** Forecasting six weeks out, you do
   not know yesterday's sales. `lag_1` in a horizon-42 model is the most common leak in
   public retail notebooks and the hardest to see, because nothing about the code looks
   wrong. Here it raises `LeakageError`.
4. **Refurbished stores vanish.** They stop appearing in the file entirely — absence, not
   a run of zeros — and produce a fake level shift on reopening. `validate.calendar_gaps`
   finds them. The synthetic generator reproduces this on purpose.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev,notebook]"
```

```bash
python -m stockout describe                          # rows, stores, nulls, calendar gaps
python -m stockout backtest --model seasonal_naive   # the baseline
python -m stockout backtest --model naive_last       # what ignoring the weekday costs
python -m stockout backtest --model gbm               # the gradient-boosted model
python -m stockout calibration                       # does each quantile cover what it claims
python -m stockout frontier --store 3                # the cost of each service level
python -m stockout frontier --model gbm_conformal    # the same, with calibrated quantiles
python -m stockout frontier --lead-time 7            # and with stock that takes a week
python -m stockout synth --out data/mine.csv --stores 20 --days 1095
```

Useful flags: `--horizon`, `--folds`, `--gap`, `--min-train-days`, and `--sliding` for a
fixed-width training window instead of an expanding one. `frontier` also takes `--model`
and `--lead-time`; a lead time above zero switches the simulator from a repeated
newsvendor to an `(R, S)` system and re-sizes the level to match.

The `gbm` models need LightGBM: `pip install -e ".[gbm]"`. Everything else runs without it,
and asking for a model you have not installed prints one line saying so rather than a
traceback.

## Data

Everything above runs offline, on a committed synthetic sample. For the real thing:

```bash
python -m stockout fetch
python -m stockout backtest --data data/raw/train.csv --model seasonal_naive
```

`fetch` needs **two** things, and the second is the one people miss:

1. A Kaggle API token at `~/.kaggle/kaggle.json` — kaggle.com → Settings → API → *Create
   New Token*.
2. Acceptance of the [competition rules](https://www.kaggle.com/competitions/rossmann-store-sales/rules),
   signed in, on the web page. **Until that is done the API returns 403 for a token that
   is otherwise perfectly valid.** The downloader says so rather than surfacing a raw
   traceback.

The archive is never committed — see [ADR 0001](docs/decisions/0001-data-is-downloaded-not-committed.md).

## Layout

```
src/stockout/
├── data/schemas.py     every assumption about the file's shape, in one place
├── data/synth.py       deterministic generator — the whole test suite runs on it
├── data/validate.py    schema guard: dtypes, unique keys, open/sales invariant
├── features/lags.py    leakage guard: shift before you roll, lag >= horizon
├── features/build.py   the model matrix, and the denylist
├── split/rolling.py    rolling origin only — there is no shuffled splitter
├── evaluate/metrics.py WMAPE, MASE, RMSPE, pinball, coverage. Deliberately no MAPE
├── evaluate/backtest.py the fold loop; a fresh model per fold
├── models/baselines.py naive_last, seasonal_naive, moving_average
├── models/gbm.py       LightGBM point (tweedie) and quantile; imported lazily
├── models/conformal.py split-conformal calibration, so a stated level means something
├── inventory/policy.py the critical ratio, the base-stock level, the order
├── inventory/simulate.py the day-by-day walk, the delivery pipeline, the frontier
└── cli.py              fetch | synth | describe | backtest | calibration | frontier

docs/questions.md       the five hypotheses, written first
docs/results.md         the answers — empty until the analysis is done
docs/decisions/         ten ADRs, each stating what the decision cost
notebooks/              the analysis
data/                   gitignored, except one small synthetic sample
reports/                generated charts — regenerated, never committed
```

## Testing

```bash
ruff check .                                  # lint only; never `ruff format` (ADR 0004)
pyright                                       # type gate
pytest --cov --cov-fail-under=90              # 326 passing, 0 skipped, 98% covered
```

Unit tests never touch the network. A `conftest.py` autouse fixture replaces
`socket.connect`, so a test that reaches for a socket fails loudly instead of passing on a
laptop that happens to have credentials and failing in CI. Network work goes behind
`@pytest.mark.integration`, excluded by default.

The ten tests that used to be skipped were the specification for the unbuilt half. They
are green now, and one of them settled a design question the prose had got wrong: the
stub's docstring described a delivery lead time inside the simulator, while its own test
required that a shortfall on one day not carry into the next. Only one of those could
survive, and the executable one won —
[ADR 0008](docs/decisions/0008-the-simulator-has-no-shipping-lag.md).

`tests/test_no_random_splits.py` parses every module's AST and asserts that no shuffled
split, no `random_state`, and no scikit-learn import executes anywhere in the package. It
parses rather than greps because a text search cannot tell code from the docstrings
explaining why the code is absent — and would therefore be defeated by deleting the
explanation.

## Known limits

- **The committed sample is synthetic.** It reproduces four structures that make the real
  problem hard, and nothing else. Anything learned from it is a statement about a
  generator, not about retail. It is used to test machinery, never to support a finding.
- **Rossmann is revenue, not units, and has no inventory column at all.** Demand and stock
  are both in currency units of stock-at-cost — internally consistent, and explicitly *not*
  a unit-level simulation. Converting via an assumed basket size would add decimal places
  and no truth. M5 is the upgrade path.
- **Pooled, but only by accident.** The baselines are strictly per store. The
  gradient-boosted models are one global fit with `store` as a feature, which pools by
  default and shares nothing deliberately — no hierarchy, no per-store effects, no
  borrowing toward a group mean. Short histories are served worst either way, and
  question Q2 exists to measure how much.
- **The quantile models still under-cover, even calibrated.** A nominal 0.9 delivers 0.721
  raw and 0.779 after conformal correction. Half the gap is gone; the other half is the
  distribution shift between a 42-day calibration window and the 42-day window after it,
  and no amount of arithmetic on the first will reveal the second.
- **Calibration needs rows and is not free below about a hundred of them.** On a two-store
  draw it made both coverage and pinball loss worse. The row count is printed rather than
  policed, because a threshold picked from four draws is a guess with a table under it.
- **Coverage is corrected on average, not per store.** A single quiet shop, or a single
  December, can still be badly covered and this layer will not notice —
  [ADR 0009](docs/decisions/0009-conformal-calibration-not-a-recalibrated-loss.md).
- **The delivery pipeline is opt-in, and stock in transit is free.** By default the
  simulator prices a repeated single-period newsvendor: stock is topped up daily, unmet
  demand is lost ([ADR 0008](docs/decisions/0008-the-simulator-has-no-shipping-lag.md)).
  `--lead-time` opens a real pipeline, but nothing is charged for goods on a lorry, so the
  model prefers a long pipeline to a full shelf in a way a financed business would not —
  [ADR 0010](docs/decisions/0010-the-pipeline-is-opt-in-and-the-critical-ratio-does-not-survive-it.md).

## Documentation

- [docs/questions.md](docs/questions.md) — the five hypotheses, committed before analysis
- [docs/results.md](docs/results.md) — the answers, and what didn't work
- [docs/architecture.md](docs/architecture.md) — flow, the three guards, module map
- [docs/data-dictionary.md](docs/data-dictionary.md) — columns, traps, future-known vs observed
- [docs/glossary.md](docs/glossary.md) — forecasting and inventory terms
- [docs/decisions/](docs/decisions/) — ten ADRs, each with its cost
- [MODEL_CARD.md](MODEL_CARD.md) — what the models are, and are not, for

## Licence

MIT — see [LICENSE](LICENSE). The Rossmann dataset is not redistributed here and remains
subject to the competition's own terms.
