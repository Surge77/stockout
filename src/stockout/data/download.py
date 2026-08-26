"""Fetch a competition archive into `data/raw/`.

Kaggle forbids redistributing competition data, so it is downloaded per machine and never
committed. Credentials come from `~/.kaggle/kaggle.json` or from `KAGGLE_USERNAME` and
`KAGGLE_KEY`; this module never reads, stores or logs them itself.

The competition rules must also be accepted in a browser first. The API answers a refusal
with a 403 that says nothing useful, so that case is turned into an instruction.

Two competitions, and they collide. Both Rossmann and Store Item Demand unpack a file
called `train.csv`, so downloading the second on top of the first would silently replace
one dataset with the other and every score after it would describe the wrong data. The
product competition therefore unpacks into its own subdirectory. Rossmann keeps the flat
layout it already has on disk: moving it would break every clone that has already fetched
it, in order to tidy a path.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import RAW_DIR
from ..errors import StockoutError

_FORBIDDEN = "403"


@dataclass(frozen=True)
class Competition:
    """One downloadable source: what to ask Kaggle for, and what it should leave behind."""

    dataset: str
    slug: str
    produces: tuple[str, ...]

    #: Where the archive unpacks, relative to `RAW_DIR`. Empty for Rossmann, which was
    #: here first and whose flat layout is already on disk in every existing clone.
    subdir: str = ""

    @property
    def rules_url(self) -> str:
        return f"https://www.kaggle.com/competitions/{self.slug}/rules"

    def destination(self, root: Path) -> Path:
        return root / self.subdir if self.subdir else root


COMPETITIONS: tuple[Competition, ...] = (
    Competition("rossmann", "rossmann-store-sales", ("train.csv", "store.csv")),
    Competition(
        "store_item_demand",
        "demand-forecasting-kernels-only",
        ("train.csv",),
        subdir="store-item-demand",
    ),
)

DOWNLOADABLE: tuple[str, ...] = tuple(c.dataset for c in COMPETITIONS)


def competition_for(dataset: str) -> Competition:
    for competition in COMPETITIONS:
        if competition.dataset == dataset:
            return competition
    offered = ", ".join(DOWNLOADABLE)
    raise StockoutError(f"{dataset!r} is not downloadable; choose from {offered}")


def _api():  # type: ignore[no-untyped-def]
    """Authenticate lazily, so importing this module needs no credentials."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise StockoutError(
            "the kaggle package is not installed; `pip install -e .` brings it in"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as exc:
        raise StockoutError(
            "no Kaggle credentials. Put kaggle.json in ~/.kaggle/, or set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc
    return api


def _unpack(destination: Path) -> None:
    for archive in destination.glob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
        archive.unlink()


def fetch(
    dataset: str = "rossmann", *, root: Path = RAW_DIR, force: bool = False
) -> list[Path]:
    """Download one competition's archive and return the files it left behind."""
    competition = competition_for(dataset)
    destination = competition.destination(root)
    destination.mkdir(parents=True, exist_ok=True)

    expected = [destination / produced for produced in competition.produces]
    if not force and all(path.exists() for path in expected):
        return expected

    api = _api()
    try:
        api.competition_download_files(competition.slug, path=str(destination), quiet=False)
    except Exception as exc:
        if _FORBIDDEN in str(exc):
            raise StockoutError(
                f"Kaggle refused {competition.slug}. Accept the rules at "
                f"{competition.rules_url} while signed in, then run this again."
            ) from exc
        raise StockoutError(f"could not download {competition.slug}: {exc}") from exc

    _unpack(destination)
    still_missing = [path.name for path in expected if not path.exists()]
    if still_missing:
        raise StockoutError(
            f"{competition.slug} downloaded but {', '.join(still_missing)} is not in "
            f"{destination}; the archive layout may have changed."
        )
    return expected
