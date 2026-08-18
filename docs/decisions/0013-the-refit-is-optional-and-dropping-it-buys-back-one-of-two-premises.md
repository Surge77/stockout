# 0013 — The refit is optional, and dropping it buys back one of two missing premises

> **Superseded by [ADR 0014](0014-a-scikit-learn-comparison-not-an-inventory-system.md).**
> The code this records — the quantile models, the conformal calibration and the
> inventory simulator — was removed in `c7fadcb`. The record is kept because a
> decision log that is edited to match the present is not a log; nothing below
> describes code that ships today.

## Situation

ADR 0009 gave up the split-conformal theorem and said exactly why:

> **Two models are fitted, not one — and this is what costs the guarantee.** [...] The
> alternative — serving predictions from the probe — would keep the theorem intact and
> reintroduce the misalignment the two-fit design exists to avoid: the probe's retained
> history stops 42 days before the test window, and a positional lag reaching across that
> hole lands on the wrong date. A silently wrong feature is worse than an unproven bound,
> so the bound is the thing given up.

That reasoning was sound and its premise was wrong. The misalignment is real — a test row's
`lag_42` lands *inside* the calibration window, so a model retaining only the inner window
would shift 42 rows across a 42-day hole and read dates roughly 84 days earlier. But that is
a fact about which rows a model keeps for **building features**, not about which rows it
**trains on**, and the two had been assumed to be the same set because nothing had ever
needed them to differ.

## Decision

`design.GbmDesign._training_matrix` takes an optional `history`. Features are built from
`history` and retained for prediction; only the rows also present in `train` become boosting
rows. `GbmQuantileForecaster.fit_within(train, history=...)` exposes it, and
`ConformalQuantileForecaster(refit=False)` uses it: the probe boosts on the inner window
while keeping the whole training frame to lag against, and is then served directly.

So the model that goes out has never trained on the calibration window, and still has a
complete calendar behind it. The offsets describe the estimator that produced them, and the
split-conformal theorem applies to that estimator.

**It does not follow that coverage is proven, and nothing in this repository will say that
it is.** The theorem needs two things and this recovers one of them. The other is
exchangeability between the calibration rows and the test rows, and time-ordered retail
demand does not have it: the test window comes *after* the calibration window, the
distribution has moved between them — that shift is the whole reason ADR 0009's coverage gap
survived calibration — and same-day rows across stores are correlated rather than
independent draws. What changed is that the refit is no longer *also* in the way. There was
one unprovable assumption stacked on one avoidable modelling choice; the choice is gone and
the assumption is named. `stockout calibration --no-refit` prints that sentence rather than
a guarantee.

**The default stays `refit=True`.** Two reasons, one weak and one strong. The weak one is
that every number this repository has published was produced with the refit. The strong one
is below: the measured difference runs in favour of the no-refit variant on the levels that
matter, and by less than the noise on 140 test rows. Changing a default on a 1.4-sigma
result is the same error as picking a threshold from four draws, which ADR 0009 already
refused.

## Consequence — ADR 0009 expected a worse trade and got a better one

`python -m stockout calibration` against `--no-refit`, newest fold, 140 held-out trading
rows, coverage gap and pinball loss per level:

| nominal | gap, refit | gap, no refit | pinball, refit | pinball, no refit |
|---|---|---|---|---|
| 0.50 | −0.086 | **−0.107** | 337.2 | **353.6** |
| 0.75 | −0.100 | −0.107 | 279.6 | 279.8 |
| 0.80 | −0.107 | **−0.079** | 250.3 | 252.2 |
| 0.90 | −0.121 | **−0.086** | 174.9 | **160.8** |
| 0.95 | −0.064 | **−0.057** | 116.1 | **104.8** |
| 0.99 | −0.061 | **−0.026** | 42.3 | **36.1** |

The worst miss moves from −0.121 at the 0.90 to −0.107 at the 0.50. **Coverage improves at
every level from the 0.80 up and degrades at the median**, which is the half of the
distribution service levels are chosen from and the half they are not. Pinball follows the
same split: better at 0.90, 0.95 and 0.99, worse at the median.

The point forecast is where the lost data shows up, and it shows up consistently:

| | WMAPE | MAE | median pinball |
|---|---|---|---|
| refit | 0.0723 | 674.4 | 337.2 |
| no refit | 0.0758 | 707.2 | 353.6 |

All three degrade by about **5%**, which is what dropping the most recent 42 days from a
roughly 600-day training window buys. That the three agree to within a tenth of a percentage
point is the check that this is a data-volume effect rather than something stranger.

Read together: giving up the refit costs about 5% of point accuracy and buys better
calibration in the upper tail plus a theorem that applies to the served estimator. ADR 0009
predicted "a worse trade". On this data it is a better one everywhere except the median.

## Cost

**The differences are inside the noise, and the table above must not be read as a result.**
Coverage measured on 140 rows has a standard error near 0.025 at the 0.9, so −0.121 against
−0.086 is about 1.4 sigma. It is suggestive and it is one fold of one synthetic generator.
The 5% accuracy cost is the only number here that is a straightforward consequence of
arithmetic rather than of a draw.

**Fitting is halved and that is not free either.** Only one set of six boosters is trained
instead of two, so `refit=False` is about twice as fast — which sounds like a benefit and is
the same fact as "the deployed model has seen less data".

**The theorem covers marginal coverage and nothing else.** ADR 0012's conditional gaps are
untouched by this: a per-store miss is not a marginal miss, and no amount of exchangeability
would make a pooled bound say anything about a particular shop.

**`fit_within` exists on the quantile forecaster only.** `GbmForecaster` has no caller that
needs it and does not get it. The `HistoryAwareQuantileModel` protocol is runtime-checkable
so that asking for `refit=False` with a model that cannot serve it fails with a sentence
naming the missing method rather than an `AttributeError` three frames down.

**Two modes now exist behind one class name.** `gbm_conformal` means the refitting variant
in the registry and in every table published before this ADR, and `--no-refit` is the only
way to reach the other one. A reader who sees `gbm_conformal` in a report and assumes the
theorem applies to it has assumed wrong, which is the same hazard ADR 0009 created and the
reason both ADRs spend most of their length on what is *not* proven.
