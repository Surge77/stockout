# Data dictionary

Canonical column names are snake_case. Rossmann ships PascalCase; the mapping lives in
one place, `src/stockout/data/schemas.py::ROSSMANN_RENAME`, so a rename on Kaggle's side
breaks one dict rather than thirty call sites.

## Columns

| Column | Rossmann name | Dtype | Notes |
|---|---|---|---|
| `date` | `Date` | `datetime64[ns]` | Daily. One row per store per trading day |
| `store` | `Store` | `int32` | 1–1115 in Rossmann; 1–N in synthetic data |
| `day_of_week` | `DayOfWeek` | `int8` | **1 = Monday … 7 = Sunday.** Not pandas' 0–6 |
| `sales` | `Sales` | `float64` | The target. Revenue for the day, not units |
| `customers` | `Customers` | `float64` | **Denylisted — see below** |
| `open` | `Open` | `int8` | 0 or 1. `open == 0` implies `sales == 0`, enforced |
| `promo` | `Promo` | `int8` | Promotion running that day. Known in advance |
| `state_holiday` | `StateHoliday` | `string` | `"0"`, `"a"`, `"b"`, `"c"`. See the dtype trap |
| `school_holiday` | `SchoolHoliday` | `int8` | Known in advance |

## The four traps

**1. `open == 0` means `sales == 0`.** Roughly a seventh of Rossmann rows are closures,
mostly Sundays. They are perfectly predictable, so the backtest excludes them from scoring.
Note that this flatters `mae` and `rmse` but **not** `wmape`, which cancels them exactly —
both behaviours are asserted in `tests/test_backtest.py`.

**2. `customers` is a leak.** It is the best single predictor of `sales` — correlation
around 0.9 — and it is *unknown at the forecast origin*. Nobody knows how many people will
walk in six weeks from now. It sits in `schemas.UNAVAILABLE_AT_FORECAST_TIME` and is
excluded by `features.build.FEATURE_DENYLIST`. A model that uses it reports a wonderful
score and cannot be deployed for a single day.

**3. `state_holiday` mixes types.** Rossmann's `train.csv` genuinely contains both the
integer `0` and the string `"0"` in this column. Pandas infers `object`, and every
`== "0"` comparison then silently misses half the rows. `loaders.canonicalise` casts
through `str` before the nullable string dtype so both land on the same value.

**4. Refurbishment gaps.** A subset of stores closes for months and simply **stops
appearing in the file** — absence, not a run of zeros. On reopening there is a level shift
that looks like signal and is not. `validate.calendar_gaps` finds them; quantify the extent
during EDA before trusting any store-level result. The synthetic generator reproduces this
deliberately.

## Future-known vs observed

The distinction that decides whether a column may be a feature:

| Kind | Examples | Usable? |
|---|---|---|
| **Future-known covariate** | `promo`, `state_holiday`, `school_holiday`, `open`, every calendar column | Yes — these come from a planning system, decided in advance |
| **Lagged observation** | `sales_lag_42`, `sales_roll_mean_28` | Only at lag ≥ horizon |
| **Contemporaneous observation** | `customers` | Never |

Collapsing the first two is the most common mistake in this problem. The promo calendar
for six weeks' time already exists; last Tuesday's sales, relative to a forecast origin six
weeks back, does not.

## Derived columns

Added by `features/calendar.py`, all future-known: `month`, `day_of_month`,
`week_of_year`, `day_of_year`, `year`, `days_since_start`, `is_weekend`,
`is_state_holiday`.

Added by `features/lags.py`: `sales_lag_{n}`, `sales_roll_mean_{w}`, `sales_roll_std_{w}`.
Default lags are the first four whole weeks that clear the horizon — 42, 49, 56, 63 at the
default 42-day horizon — because same-weekday history is what carries the signal.
