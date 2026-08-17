"""Turn a results frame into markdown, so a number can be pasted into the README.

Deliberately plain text rather than a plot. A chart of five folds is decoration; the
table is the evidence, and it survives being quoted in an issue or a commit message.

The table is assembled by hand rather than with `DataFrame.to_markdown`, which requires
the `tabulate` package. Twenty lines of string joining does not justify a dependency
that would then need pinning, auditing and updating forever.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_DATE_COLUMNS: tuple[str, ...] = ("train_start", "train_end", "test_start", "test_end")

_FLOAT_FORMATS: dict[str, str] = {
    "mae": "{:.1f}",
    "rmse": "{:.1f}",
    "wmape": "{:.4f}",
    "rmspe": "{:.4f}",
    "mase": "{:.3f}",
}


def to_markdown(results: pd.DataFrame, *, model_name: str) -> str:
    """Per-fold table plus a verdict line, as a markdown fragment."""
    if results.empty:
        return f"No folds were scored for `{model_name}`."

    display = results.copy()
    for column in _DATE_COLUMNS:
        if column in display.columns:
            display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d")
    for column, fmt in _FLOAT_FORMATS.items():
        if column in display.columns:
            display[column] = display[column].map(lambda v, f=fmt: f.format(v))

    header = f"### `{model_name}` — {len(results)} rolling-origin folds"
    table = _markdown_table(
        [str(c) for c in display.columns],
        [[str(v) for v in row] for row in display.itertuples(index=False)],
    )
    return f"{header}\n\n{table}\n\n{_verdict_line(results)}\n"


_FRONTIER_FORMATS: dict[str, str] = {
    "quantile": "{:.2f}",
    "fill_rate": "{:.4f}",
    "cycle_service_level": "{:.3f}",
    "holding_cost": "{:,.0f}",
    "transit_cost": "{:,.0f}",
    "shortage_cost": "{:,.0f}",
    "total_cost": "{:,.0f}",
    "mean_on_hand": "{:,.0f}",
    "mean_on_order": "{:,.0f}",
}

_CALIBRATION_FORMATS: dict[str, str] = {
    "quantile": "{:.2f}",
    "empirical": "{:.3f}",
    "gap": "{:+.3f}",
    "pinball": "{:,.1f}",
}


def frontier_to_markdown(table: pd.DataFrame) -> str:
    """The cost of each service level, and which one was cheapest.

    Naming the winner is the whole point. A frontier table that leaves the reader to
    scan for the smallest number is a chart pretending to be an argument.
    """
    if table.empty:
        return "No service levels were priced."

    display = table.copy()
    for column, fmt in _FRONTIER_FORMATS.items():
        if column in display.columns:
            display[column] = display[column].map(lambda v, f=fmt: f.format(v))

    header = f"### the cost of stocking to each quantile — {len(table)} levels priced"
    body = _markdown_table(
        [str(c) for c in display.columns],
        [[str(v) for v in row] for row in display.itertuples(index=False)],
    )
    return f"{header}\n\n{body}\n\n{_cheapest_line(table)}\n"


def calibration_to_markdown(table: pd.DataFrame, *, model_name: str) -> str:
    """Nominal against empirical coverage, and the worst miss named outright.

    The single number a reader should leave with is the largest gap, because a frontier
    built on levels that do not hold prices a policy nobody selected. Naming it is the
    same discipline as `frontier_to_markdown` naming the cheapest row.
    """
    if table.empty:
        return f"No quantiles were scored for `{model_name}`."

    display = table.copy()
    for column, fmt in _CALIBRATION_FORMATS.items():
        if column in display.columns:
            display[column] = display[column].map(lambda v, f=fmt: f.format(v))

    header = f"### `{model_name}` — coverage against the level it claims"
    body = _markdown_table(
        [str(c) for c in display.columns],
        [[str(v) for v in row] for row in display.itertuples(index=False)],
    )
    return f"{header}\n\n{body}\n\n{_worst_miss_line(table)}\n"


def _worst_miss_line(table: pd.DataFrame) -> str:
    """Positional, and signed: under-covering is the direction that costs a sale.

    A NaN gap is not a small gap. `coverage_table` produces one for a column whose label
    is not a quantile, and for a scoring window with no rows in it; `argmax` over an array
    containing NaN still returns an index, and the line would then announce a miss of
    `nan` as over-coverage. Non-finite rows are dropped, and a table with nothing finite
    left says so rather than naming one.
    """
    gaps = table["gap"].to_numpy(dtype="float64")
    finite = np.flatnonzero(np.isfinite(gaps))
    if finite.size == 0:
        return "**No level could be scored** — no finite coverage gap in the table."

    worst = int(finite[np.abs(gaps[finite]).argmax()])
    quantile = float(table["quantile"].to_numpy(dtype="float64")[worst])
    empirical = float(table["empirical"].to_numpy(dtype="float64")[worst])
    direction = "under-covers" if gaps[worst] < 0 else "over-covers"
    return (
        f"**Worst miss at quantile {quantile:.2f}** — {direction} by "
        f"{abs(gaps[worst]):.3f}, delivering {empirical:.1%} of the days it promises."
    )


def _cheapest_line(table: pd.DataFrame) -> str:
    """Positional rather than label-based: a frontier's index carries no meaning."""
    costs = table["total_cost"].to_numpy(dtype="float64")
    best = int(costs.argmin())
    quantile = float(table["quantile"].to_numpy(dtype="float64")[best])
    fill_rate = float(table["fill_rate"].to_numpy(dtype="float64")[best])
    short_days = int(table["stockout_days"].to_numpy(dtype="int64")[best])
    return (
        f"**Cheapest at quantile {quantile:.2f}** — total cost {costs[best]:,.0f}, "
        f"fill rate {fill_rate:.1%}, {short_days} short day(s)."
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]

    def line(cells: Sequence[str]) -> str:
        padded = [cell.ljust(width) for cell, width in zip(cells, widths, strict=True)]
        return "| " + " | ".join(padded) + " |"

    rule = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), rule, *(line(row) for row in rows)])


def _verdict_line(results: pd.DataFrame) -> str:
    mean_wmape = float(results["wmape"].mean())
    mean_mase = float(results["mase"].mean())
    return f"**Mean WMAPE {mean_wmape:.4f} · MASE {mean_mase:.3f}** — {_verdict(mean_mase)}"


def _verdict(mean_mase: float) -> str:
    """State plainly whether the baseline was beaten. No hedging in a results table."""
    if pd.isna(mean_mase):
        return "MASE undefined (the baseline made no errors to scale by)."
    if abs(mean_mase - 1.0) < 1e-9:
        return "this *is* the seasonal-naive baseline."
    if mean_mase < 1.0:
        return f"beats seasonal-naive by {(1.0 - mean_mase) * 100:.1f}%."
    return f"**loses** to seasonal-naive by {(mean_mase - 1.0) * 100:.1f}%."
