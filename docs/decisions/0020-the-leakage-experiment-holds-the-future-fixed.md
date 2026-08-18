# 0020 — The leakage experiment holds the future fixed

## Situation

The obvious way to show that a random split leaks is to score one and score a time split
and compare. It does not work, and the reason is worth stating because the experiment is
so tempting.

Those two protocols differ in **three** things at once: how many rows each trains on, which
stretch of calendar each is tested on, and whether either leaks. The gap between their
scores cannot be attributed to any one of the three. An examiner who notices has ended the
discussion, and they will notice, because it is the first thing to check.

## Decision

The test window is **held fixed**. The last 28 days of the calendar are removed once, with
a horizon-sized gap in front of them, and every arm is scored on that same window. No arm
trains on it.

What varies between arms is only how each selects its training and validation rows out of
everything *before* that window — and therefore only what each arm **believes about
itself**.

Every arm reports three numbers:

- `internal` — what the protocol would have told you.
- `future` — what that same model then scored on the fixed window.
- `optimism` — the difference, and the column the module exists to produce.

That reframes the claim from *"this model is better"* to *"this protocol's estimate of its
own error was wrong by this much"*, which is both the true statement and the more useful
one.

One estimator throughout — `Ridge(alpha=1.0)` — because the comparison is between
protocols, and changing the model as well would confound it again.

Four arms, on the committed sample (R², 28-day window, 7-day gap):

| arm | protocol | features | internal | future | optimism |
|---|---|---|---|---|---|
| honest | time-ordered | denylist enforced | 0.9013 | 0.8447 | 0.0566 |
| shuffled split | random | denylist enforced | 0.9131 | 0.8413 | 0.0718 |
| preprocessing leak | time-ordered | scaler and imputer fitted on all rows | 0.9014 | 0.8447 | 0.0567 |
| future feature | time-ordered | `customers` smuggled past the denylist | 0.9984 | 0.9988 | **-0.0004** |

## What it found, including the part that is unflattering

**The shuffled split's optimism is 0.0718 against the honest arm's 0.0566.** A gap of
0.015 R². That is smaller than this project has previously implied a shuffled split costs,
and the honest reading is that on *this generator* it barely matters — the synthetic
promotion calendar and weekday pattern are deterministic, so a Tuesday in training tells
you little about the Wednesday beside it that the calendar features have not already told
you. On real data with genuine local autocorrelation the gap is expected to be larger. It
is not measured here, because the real file needs a Kaggle account.

**The preprocessing leak found nothing at all**: 0.0567 against 0.0566, a difference in the
fourth decimal. Fitting the scaler and imputer on every row before splitting changed
essentially nothing, because a `StandardScaler`'s mean over 2,700 rows and over 2,400 of
them are the same number to three places. The leak is real, the mechanism is exactly as
described, and on this sample it is worth 0.0001. That is recorded rather than quietly
dropped from the table.

**The future feature is the one that is catastrophic.** Allowing `customers` back in — the
column Rossmann ships, which correlates with `sales` at about 0.9 and which nobody knows
in advance — takes R² to 0.9984 *and its optimism to below zero*. It scores beautifully on
both windows, and is undeployable for a single day. That is the finding worth having: a
high score on a genuinely held-out set is not evidence that a model can be used.

Getting `customers` into that arm requires renaming it, because `FEATURE_DENYLIST` refuses
it by name and cannot be switched off from outside. Needing to lie to the guard is the
strongest available evidence that the guard is load-bearing rather than decorative.

## Cost

**Three of the four arms produce a null result on the committed data.** The module's most
striking output is the arm that cheats outright, which is also the least surprising one.
A reader could reasonably conclude the leakage machinery is over-built for what it detects
here — and would be right about here, and wrong about Rossmann.

**The fixed window is one window.** Every number above rests on the same 28 days, so
anything peculiar about that stretch of December is baked into all four arms equally. A
rolling version would cost four times as many fits and is not implemented.

**`optimism` is not a standard term** and does not appear in the literature under that
name. It is `internal - future` and nothing more, and the column header is doing work that
a citation would do better.
