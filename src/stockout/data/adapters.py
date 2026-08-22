"""Where the retail data comes from, and the one shape everything downstream sees.

Two sources. `synthetic` is committed, so a fresh clone runs offline. `rossmann` is the
real competition data, which may not be redistributed and so has to be downloaded once.

Both produce the same columns — `entity, date, sales, open, promo, state_holiday,
school_holiday` — so the registry, the comparison table and the web panel are written once
rather than once per source. Adding another retail source means writing one loader here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR, SAMPLE_PATH, SAMPLE_STORES_PATH
from ..errors import StockoutError
from . import schemas as s


@dataclass(frozen=True)
class Source:
    """One dataset: how to load it, and what a reader needs to know about it."""

    name: str
    description: str
    load: Callable[[], tuple[pd.DataFrame, pd.DataFrame | None]]
    """Returns (canonical sales frame, per-entity metadata or None)."""

    entity_label: str = "store"
    raw_files: tuple[str, ...] = field(default_factory=tuple)
    origin: str = ""


def _require(paths: list[Path], source: str, how: str) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise StockoutError(
            f"{source} needs {names} under {RAW_DIR}. {how} "
            f"Meanwhile `--dataset synthetic` works offline and ships with the repository."
        )


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Common tail for both adapters: types, ordering, and the canonical column check."""
    missing = [c for c in s.CANONICAL_COLUMNS if c not in frame.columns]
    if missing:
        raise StockoutError(f"adapter produced no {', '.join(missing)} column")

    frame = frame.copy()
    frame[s.DATE] = pd.to_datetime(frame[s.DATE])
    frame[s.SALES] = pd.to_numeric(frame[s.SALES], errors="coerce").astype("float64")
    for flag in (s.OPEN, s.PROMO, s.SCHOOL_HOLIDAY):
        frame[flag] = pd.to_numeric(frame[flag], errors="coerce").fillna(0).astype("int8")
    frame[s.STATE_HOLIDAY] = frame[s.STATE_HOLIDAY].astype("string").fillna("0")
    frame[s.ENTITY] = frame[s.ENTITY].astype("string")
    return (
        frame.dropna(subset=[s.SALES])
        .sort_values([s.ENTITY, s.DATE])
        .reset_index(drop=True)
    )


def _load_synthetic() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    _require(
        [SAMPLE_PATH, SAMPLE_STORES_PATH],
        "synthetic",
        "Regenerate them with `python scripts/make_sample.py`.",
    )
    sales = pd.read_csv(SAMPLE_PATH)
    stores = pd.read_csv(SAMPLE_STORES_PATH)
    sales[s.ENTITY] = sales[s.STORE].astype("string")
    stores[s.ENTITY] = stores[s.STORE].astype("string")
    return _finalise(sales), stores


def _load_rossmann() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train, store = RAW_DIR / "train.csv", RAW_DIR / "store.csv"
    _require(
        [train, store],
        "rossmann",
        "Run `python -m stockout fetch` once, with a kaggle.json and the competition "
        "rules accepted at kaggle.com/competitions/rossmann-store-sales/rules.",
    )
    sales = pd.read_csv(train, low_memory=False).rename(columns=s.ROSSMANN_RENAME)
    stores = pd.read_csv(store).rename(columns=s.ROSSMANN_STORE_RENAME)
    sales[s.ENTITY] = sales[s.STORE].astype("string")
    stores[s.ENTITY] = stores[s.STORE].astype("string")
    return _finalise(sales), stores


SOURCES: tuple[Source, ...] = (
    Source(
        name="synthetic",
        description="Rossmann-shaped store-day sales, generated and committed.",
        load=_load_synthetic,
        origin="scripts/make_sample.py",
    ),
    Source(
        name="rossmann",
        description="Rossmann store sales: 1,115 German drugstores over 942 days.",
        load=_load_rossmann,
        raw_files=("train.csv", "store.csv"),
        origin="kaggle competitions rossmann-store-sales",
    ),
)

SOURCE_NAMES: tuple[str, ...] = tuple(source.name for source in SOURCES)


def source_for(name: str) -> Source:
    for source in SOURCES:
        if source.name == name:
            return source
    raise StockoutError(f"unknown dataset {name!r}; choose from {', '.join(SOURCE_NAMES)}")


def available(source: Source) -> bool:
    """Whether this source's files are present, without raising."""
    try:
        source.load()
    except StockoutError:
        return False
    return True
