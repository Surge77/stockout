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
python -m stockout backtest --model seasonal_naive     # a real number, no ML yet
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
construction, metrics, the backtest loop, three baselines. Everything with one correct
answer, tested and type-checked.

**In the notebook** — the analysis. The five questions are committed in
[docs/questions.md](docs/questions.md) *before* any chart exists, so findings cannot be
retrofitted to whatever turned up. See [ADR 0006](docs/decisions/0006-analysis-in-notebooks.md).

**Not built yet** — the LightGBM point and quantile models (`models/gbm.py`) and the
inventory simulator (`inventory/`) are typed stubs with their reasoning committed and
their tests written and skipped. The specification exists; the implementation does not.
The backtest harness runs without them.

## Current state

Everything below comes from `python -m stockout backtest` on the committed synthetic
sample — 4 stores, 730 days, 5 rolling-origin folds, 42-day horizon.

| Model | Mean WMAPE | MASE | |
|---|---|---|---|
| `seasonal_naive` | 0.1489 | **1.000** | the baseline, by definition |
| `moving_average` | 0.1606 | 1.080 | 8.0% worse |
| `naive_last` | 0.1695 | 1.144 | 14.4% worse |

The gap between the top and bottom rows is the value of knowing what day of the week it
is. **No model has beaten the baseline yet, because no model exists yet** — that is the
point of establishing the floor first.

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
python -m stockout synth --out data/mine.csv --stores 20 --days 1095
```

Useful flags: `--horizon`, `--folds`, `--gap`, `--min-train-days`, and `--sliding` for a
fixed-width training window instead of an expanding one.

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
├── models/gbm.py       STUB — LightGBM point (tweedie) and quantile
├── inventory/          STUB — order-up-to policy and the cost simulation
└── cli.py              stockout fetch | synth | describe | backtest

docs/questions.md       the five hypotheses, written first
docs/results.md         the answers — empty until the analysis is done
docs/decisions/         seven ADRs, each stating what the decision cost
notebooks/              the analysis
data/                   gitignored, except one small synthetic sample
reports/                generated charts — regenerated, never committed
```

## Testing

```bash
ruff check .                                  # lint only; never `ruff format` (ADR 0004)
pyright                                       # type gate
pytest --cov --cov-fail-under=90              # 110 passing, 10 skipped, 98% covered
```

Unit tests never touch the network. A `conftest.py` autouse fixture replaces
`socket.connect`, so a test that reaches for a socket fails loudly instead of passing on a
laptop that happens to have credentials and failing in CI. Network work goes behind
`@pytest.mark.integration`, excluded by default.

The ten skipped tests are the specification for the unbuilt half — the GBM models and the
inventory simulator. They are written first on purpose, so the intended behaviour is on
record before the implementation can shape it.

## Known limits

- **The committed sample is synthetic.** It reproduces four structures that make the real
  problem hard, and nothing else. Anything learned from it is a statement about a
  generator, not about retail. It is used to test machinery, never to support a finding.
- **Rossmann is revenue, not units, and has no inventory column at all.** When the
  simulator lands, demand and stock will both be in currency units of stock-at-cost —
  internally consistent, and explicitly *not* a unit-level simulation. Converting via an
  assumed basket size would add decimal places and no truth. M5 is the upgrade path.
- **No cross-series learning.** Per-store baselines cannot share structure between stores,
  which hurts short histories most. Question Q2 exists to measure how much.
- **The decision layer is unbuilt**, so the central claim — that stocking to a forecast
  quantile beats stocking to a mean — is currently a hypothesis (Q5), not a result.

## Documentation

- [docs/questions.md](docs/questions.md) — the five hypotheses, committed before analysis
- [docs/results.md](docs/results.md) — the answers, and what didn't work
- [docs/architecture.md](docs/architecture.md) — flow, the three guards, module map
- [docs/data-dictionary.md](docs/data-dictionary.md) — columns, traps, future-known vs observed
- [docs/glossary.md](docs/glossary.md) — forecasting and inventory terms
- [docs/decisions/](docs/decisions/) — seven ADRs, each with its cost
- [MODEL_CARD.md](MODEL_CARD.md) — what the models are, and are not, for

## Licence

MIT — see [LICENSE](LICENSE). The Rossmann dataset is not redistributed here and remains
subject to the competition's own terms.
