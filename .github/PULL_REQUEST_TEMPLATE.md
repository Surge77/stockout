## What changed

<!-- The behaviour, not the diff. "Bound the fold count instead of silently truncating"
     beats "edited rolling.py". -->

## Why

<!-- If this has a real trade-off, it needs an ADR in docs/decisions/ — link it here. -->

## Gates

- [ ] `ruff check .`
- [ ] `pyright`
- [ ] `pytest --cov --cov-fail-under=90`

## Leakage

<!-- Delete this whole section if the PR touches none of features/lags.py,
     features/build.py, split/rolling.py or evaluate/backtest.py. -->

- [ ] There is a test that **fails before this change and passes after**
- [ ] The sentinel test (`test_features_ignore_a_perturbed_final_target`) still passes
- [ ] Its partner (`test_perturbing_an_old_target_does_change_later_features`) still passes

A leakage regression is invisible: it makes the numbers *better*. That is why these need
saying out loud rather than trusting the suite to be green.

## Numbers

<!-- If this changes any score, paste the before and after backtest tables. A model change
     with no number attached cannot be reviewed. -->
