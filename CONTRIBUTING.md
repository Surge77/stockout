# Contributing

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev,notebook]"
```

Editable install on purpose: the tests import the package the way a user would, so there
is no `sys.path` shim and no `pythonpath` setting to drift.

## The three gates

All three must pass before a pull request is opened. CI runs exactly these.

```bash
ruff check .                          # lint only — never `ruff format`, see ADR 0004
pyright                               # type gate
pytest --cov --cov-fail-under=90      # 98% today; the gate is deliberately lower
```

## House rules

- **300 lines per file.** Split by responsibility before exceeding it, don't after.
- `from __future__ import annotations` at the top of every module.
- Type hints on every signature, private helpers included.
- Comments explain **why**, never what. A constant with a non-obvious value gets a
  sentence about where it came from.
- No new dependency for something twenty lines of standard library would do. The markdown
  table in `evaluate/report.py` is hand-rolled rather than pulling in `tabulate`.

## Tests

- Mirror the module: `src/stockout/x.py` → `tests/test_x.py`. Flat directory.
- Name the behaviour, not the function: `test_no_fold_trains_on_the_future`, not
  `test_rolling_origin_2`.
- **No network in unit tests.** A `conftest.py` autouse fixture replaces `socket.connect`,
  so an accidental request fails loudly rather than passing on a machine with credentials.
  Genuine network tests take `@pytest.mark.integration` and are excluded by default.
- `filterwarnings = ["error"]`. If a warning needs silencing, add it to the list with a
  written justification — never a bare `ignore`.
- A test that cannot fail is worse than no test. `test_features_ignore_a_perturbed_final_target`
  has a partner asserting the opposite case for exactly this reason; if you add a property
  test, add its partner too.

## Anything touching leakage

Changes to `features/lags.py`, `features/build.py` or `split/rolling.py` need a test that
**fails before the change and passes after**. These three modules are the reason the
project exists, and a regression in them is invisible — it makes the numbers better.

## Commits and branches

Conventional Commits with the module as scope, lowercase imperative subject describing the
behaviour change rather than the diff:

```
feat(split): rolling-origin splitter that refuses a shuffled fold
fix(metrics): scale MASE by the same window it is reported on
test(lags): sentinel guard against a perturbed final target
docs(adr): record why quantiles beat a mean plus a z-score
```

Branches: `feature/<kebab-slug>`, `fix/<kebab-slug>`, `test/<kebab-slug>`. Never commit
directly to `main`; everything lands through a pull request, merged with `--no-ff` so the
diff record survives.

## Decisions

Anything with a real trade-off gets an ADR in `docs/decisions/`, numbered and kebab-cased,
stating the situation, the choice, and **what it cost**. A decision with no downside listed
is usually one that was not thought about hard enough.

## Regenerating the sample

```bash
python scripts/make_sample.py
```

Only needed after changing `stockout.data.synth`. The generator is deterministic, so any
other diff means something changed that should not have.
