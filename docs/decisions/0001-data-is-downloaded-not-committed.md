# 0001 — Data is downloaded, never committed

## Situation

The Rossmann Store Sales data is a Kaggle **competition** dataset. Its rules forbid
redistribution, and the archive is tens of megabytes — the sort of thing that makes a
git history permanently heavier for no benefit.

But a repository that cannot run until someone signs up for an account elsewhere is a
repository nobody evaluates.

## Decision

Three tiers.

1. `stockout fetch` downloads the real archive per machine into `data/raw/`, which is
   gitignored.
2. A **deterministic synthetic generator** (`stockout.data.synth`) ships in the package.
   The entire test suite runs against it. It never touches the network.
3. One small synthetic sample (`data/sample_sales.csv`, ~100 KB) is committed, so a fresh
   clone can run `describe` and `backtest` and see a real number in the first minute.

## Cost

**A fresh clone cannot reproduce the headline numbers.** It reproduces the *pipeline* and
a synthetic result; the Rossmann figures need an account, an API token, and acceptance of
the competition rules on Kaggle's website. That last step is invisible — the API returns
403 for a perfectly valid token until it is done — so `download.py` carries the
instruction in its error message rather than letting a raw traceback escape.

**The synthetic data is not real.** Anything learned from it is a statement about a
generator. It is used for testing the machinery and never for a finding.
