"""Turn a results frame into markdown, so a number can be pasted into the README.

Deliberately plain text rather than a plot. A chart of five folds is decoration; the
table is the evidence, and it survives being quoted in an issue or a commit message.

The table is assembled by hand rather than with `DataFrame.to_markdown`, which requires
the `tabulate` package. Twenty lines of string joining does not justify a dependency
that would then need pinning, auditing and updating forever.

`to_markdown` knows what a backtest frame contains and formats each column accordingly.
`frame_to_markdown` knows nothing and renders whatever it is handed, which is what the
leakage and tuning tables need — their columns are named by the experiment rather than
by this module.
"""

from __future__ import annotations

from collections.abc import Sequence

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


def frame_to_markdown(frame: pd.DataFrame, *, places: int = 4) -> str:
    """Any frame as a markdown table, floats rounded and everything else left alone.

    For results whose columns this module cannot know in advance — the leakage arms and
    the tuning search. `to_markdown` above is the opinionated version, and it stays
    opinionated: a backtest's dates and its WMAPE want different formatting, and a
    renderer that guesses would get one of them wrong.
    """
    if frame.empty:
        return "_(no rows)_"

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda v, p=places: f"{v:.{p}f}")

    return _markdown_table(
        [str(c) for c in display.columns],
        [[str(v) for v in row] for row in display.itertuples(index=False)],
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
