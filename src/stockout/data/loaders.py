"""Read sales CSVs into the canonical schema, with the dtype landmines defused.

The `state_holiday` handling is not defensive coding for a case that cannot happen.
Rossmann's train.csv genuinely mixes the integer `0` and the string `"0"` in one
column, so pandas infers `object` and every downstream comparison silently misses
half the rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import schemas as s


def read_sales(path: str | Path) -> pd.DataFrame:
    """Read a sales CSV, rename to canonical columns, coerce dtypes and sort.

    Accepts either the raw Rossmann spelling (PascalCase) or a frame already written
    in canonical snake_case, so the committed sample and the downloaded archive go
    through exactly one code path.
    """
    frame = pd.read_csv(Path(path), low_memory=False)
    return canonicalise(frame)


def canonicalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename, coerce and sort an already-read frame. Pure; does not mutate input."""
    out = frame.rename(columns=s.ROSSMANN_RENAME).copy()

    if s.DATE in out.columns:
        out[s.DATE] = pd.to_datetime(out[s.DATE], errors="raise")

    if s.STATE_HOLIDAY in out.columns:
        # Cast through str before the nullable string dtype so an integer 0 and a
        # string "0" land on the same value instead of two distinct categories.
        out[s.STATE_HOLIDAY] = out[s.STATE_HOLIDAY].astype(str).astype("string")

    for column, dtype in s.DTYPES.items():
        if column in out.columns and column != s.STATE_HOLIDAY:
            # Resolve the string spelling to a real dtype object. `.astype("int8")` works
            # at runtime but is opaque to a type checker, and the schema is worth keeping
            # as readable strings rather than imported dtype singletons.
            out[column] = out[column].astype(pd.api.types.pandas_dtype(dtype))

    sort_keys = [c for c in s.KEY_COLUMNS if c in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys).reset_index(drop=True)
    return out


def write_sales(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a sales frame as CSV in canonical spelling. Returns the path written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, date_format="%Y-%m-%d")
    return destination
