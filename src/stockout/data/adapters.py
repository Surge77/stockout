"""Where the retail data comes from, and the one shape everything downstream sees.

Four sources in two shapes. `synthetic` and `synthetic_products` are committed, so a fresh
clone runs offline; `rossmann` and `store_item_demand` are the real competition data, which
may not be redistributed and so have to be downloaded once each.

**Store-day sources** answer *what will this shop take on this day*. One row per shop-day,
`sales` in currency, and a metadata file describing each shop.

**Product sources** answer *how many of this item will this shop shift on this day*. One
row per shop-item-day, `sales` in units, and no metadata at all — Kaggle's Store Item
Demand data publishes nothing about what any item is. The `entity` is the store-item pair.

Every source produces the same canonical columns, so the registry, the comparison table and
the web panel are written once rather than once per shape. What differs between the two is
declared on `Source` rather than inferred from the dataset name downstream: a product source
has no promotions to offer in a form and no store attributes to describe a hypothetical
shop with, and the surfaces read those flags instead of guessing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import (
    RAW_DIR,
    SAMPLE_CATALOGUE_PATH,
    SAMPLE_PATH,
    SAMPLE_PRODUCTS_PATH,
    SAMPLE_STORES_PATH,
)
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

    #: What `sales` counts. Currency for a till total, units for a product count. Drives
    #: wording only — no arithmetic anywhere reads it.
    unit: str = "currency"

    #: Whether `entity` is a store-item pair rather than a store. When set, the adapter
    #: also emits `store` and `item`, which `features` turns into two separate codes so a
    #: model can learn item levels and store multipliers independently.
    has_products: bool = False

    #: Whether the source records promotions and holidays. False on the product sources,
    #: whose rows carry none: the adapter supplies the canonical flags as constants and the
    #: form hides the inputs rather than offering controls that cannot change an answer.
    has_promotions: bool = True

    #: Whether a per-entity metadata file exists. False on the product sources, which is
    #: what makes the described-shop page unanswerable there — see `hypothetical.py`.
    has_store_metadata: bool = True

    #: Whether the source says what its items *are*. True only for `synthetic_products`,
    #: which this repository generates and can therefore give a catalogue; the real Store
    #: Item Demand file publishes an integer per item and nothing else, so an item there is
    #: its number and its history and there is no honest name to print beside it.
    has_item_metadata: bool = False

    @property
    def sample_hint(self) -> str:
        """The committed source to fall back to, named by shape rather than guessed."""
        return "synthetic_products" if self.has_products else "synthetic"


def _require(paths: list[Path], source: str, how: str, *, fallback: str = "synthetic") -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        where = missing[0].parent
        raise StockoutError(
            f"{source} needs {names} under {where}. {how} "
            f"Meanwhile `--dataset {fallback}` works offline and ships with the repository."
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


#: Store Item Demand unpacks its own `train.csv`, and so does Rossmann. Both landing in
#: `data/raw/` would have the second download silently overwrite the first, so the product
#: competition gets a subdirectory. Rossmann keeps the flat layout it already has on disk:
#: moving it would break every clone that has already fetched it, to tidy a path.
PRODUCT_RAW_DIR = RAW_DIR / "store-item-demand"


def _as_product_panel(sales: pd.DataFrame) -> pd.DataFrame:
    """Give a store-item frame the canonical columns, and say what it does not have.

    The four flags are constants here because the source genuinely carries none of them:
    every day is a trading day, and no promotion or holiday is recorded against any row.
    Supplying them as zeros keeps one canonical shape for every source; `Source` declares
    the absence so no surface mistakes a constant column for a real one.
    """
    frame = sales.copy()
    frame[s.ENTITY] = (
        frame[s.STORE].astype("string")
        + s.PRODUCT_ENTITY_SEPARATOR
        + frame[s.ITEM].astype("string")
    )
    frame[s.OPEN] = 1
    frame[s.PROMO] = 0
    frame[s.STATE_HOLIDAY] = "0"
    frame[s.SCHOOL_HOLIDAY] = 0
    return frame


def _catalogue_per_entity(sales: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """The item catalogue, restated one row per store-item pair.

    Metadata is joined on `entity` everywhere in this package, and the catalogue is per
    *item* — the same description belongs to store 3's copy and store 8's. Expanding it
    here keeps one join in `features` rather than a second keyed on a different column.
    """
    entities = sales[[s.ENTITY, s.ITEM]].drop_duplicates()
    joined = entities.merge(catalogue, on=s.ITEM, how="left")
    joined[s.ENTITY] = joined[s.ENTITY].astype("string")
    return joined


def _load_synthetic_products() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    _require(
        [SAMPLE_PRODUCTS_PATH, SAMPLE_CATALOGUE_PATH],
        "synthetic_products",
        "Regenerate them with `python scripts/make_product_sample.py`.",
        fallback="synthetic_products",
    )
    sales = _finalise(_as_product_panel(pd.read_csv(SAMPLE_PRODUCTS_PATH)))
    catalogue = pd.read_csv(SAMPLE_CATALOGUE_PATH)
    return sales, _catalogue_per_entity(sales, catalogue)


def _load_store_item_demand() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train = PRODUCT_RAW_DIR / "train.csv"
    _require(
        [train],
        "store_item_demand",
        "Run `python -m stockout fetch --dataset store_item_demand` once, with a "
        "kaggle.json and the competition rules accepted at "
        "kaggle.com/competitions/demand-forecasting-kernels-only/rules.",
        fallback="synthetic_products",
    )
    sales = pd.read_csv(train).rename(columns=s.STORE_ITEM_RENAME)
    return _finalise(_as_product_panel(sales)), None


SOURCES: tuple[Source, ...] = (
    Source(
        name="synthetic",
        description="Rossmann-shaped store-day sales, generated and committed.",
        load=_load_synthetic,
        origin="scripts/make_sample.py",
    ),
    Source(
        name="rossmann",
        description="Rossmann store sales: 1,115 German health-and-beauty shops over 942 days.",
        load=_load_rossmann,
        raw_files=("train.csv", "store.csv"),
        origin="kaggle competitions rossmann-store-sales",
    ),
    Source(
        name="synthetic_products",
        description="Store-item unit demand, generated and committed.",
        load=_load_synthetic_products,
        entity_label="store-item",
        origin="scripts/make_product_sample.py",
        unit="units",
        has_products=True,
        has_promotions=False,
        has_store_metadata=False,
        has_item_metadata=True,
    ),
    Source(
        name="store_item_demand",
        description="Store Item Demand: 10 shops x 50 items of daily unit sales over five years.",
        load=_load_store_item_demand,
        entity_label="store-item",
        raw_files=("train.csv",),
        origin="kaggle competitions demand-forecasting-kernels-only",
        unit="units",
        has_products=True,
        has_promotions=False,
        has_store_metadata=False,
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
