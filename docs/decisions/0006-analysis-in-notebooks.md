# 0006 — Analysis in notebooks, plumbing in the package

## Situation

Two failure modes, opposite directions. Put everything in notebooks and nothing is tested,
nothing is reusable, and the repository is a pile of JSON. Put everything in the package
and the analytical choices — which stores to group, which window to smooth over, where to
cut an axis — get buried inside functions where a reader cannot see them next to the chart
they produced.

## Decision

The dividing line is **whether there is one correct answer**.

**In the package** (`src/stockout/`, tested, type-checked): acquisition, schema
validation, cleaning, splitting, feature construction, metrics, the backtest loop,
baselines. Every one of these has a right answer that a test can assert.

**In the notebook** (`notebooks/01_explore.ipynb`, reviewed by eye): the analysis. Which
comparison to draw, which aggregation makes the point, what the chart means. These are
judgements, and they belong visible beside their output.

The conclusions are then written in prose in [results.md](../results.md), so the finding
survives without anyone re-running a kernel.

## Cost

**Notebook code is untested and uncovered.** A bug in an aggregation there will not be
caught by CI. The mitigation is that anything reused twice gets promoted into the package,
where it acquires a test on the way in.

**Notebooks diff badly.** `.gitattributes` marks them `linguist-documentation` so they do
not distort the repository's language statistics, but a meaningful review of a notebook
diff is still hard. Keeping them few and small is the only real defence.
