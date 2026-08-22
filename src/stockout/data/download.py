"""Fetch the Rossmann archive into `data/raw/`.

Kaggle forbids redistributing competition data, so it is downloaded per machine and never
committed. Credentials come from `~/.kaggle/kaggle.json` or from `KAGGLE_USERNAME` and
`KAGGLE_KEY`; this module never reads, stores or logs them itself.

The competition rules must also be accepted in a browser first. The API answers a refusal
with a 403 that says nothing useful, so that case is turned into an instruction.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from ..config import RAW_DIR
from ..errors import StockoutError

COMPETITION = "rossmann-store-sales"
RULES_URL = f"https://www.kaggle.com/competitions/{COMPETITION}/rules"
PRODUCES: tuple[str, ...] = ("train.csv", "store.csv")

_FORBIDDEN = "403"


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


def fetch(*, destination: Path = RAW_DIR, force: bool = False) -> list[Path]:
    """Download the archive and return the files it left behind."""
    destination.mkdir(parents=True, exist_ok=True)
    expected = [destination / produced for produced in PRODUCES]
    if not force and all(path.exists() for path in expected):
        return expected

    api = _api()
    try:
        api.competition_download_files(COMPETITION, path=str(destination), quiet=False)
    except Exception as exc:
        if _FORBIDDEN in str(exc):
            raise StockoutError(
                f"Kaggle refused {COMPETITION}. Accept the rules at {RULES_URL} while "
                f"signed in, then run this again."
            ) from exc
        raise StockoutError(f"could not download {COMPETITION}: {exc}") from exc

    _unpack(destination)
    still_missing = [path.name for path in expected if not path.exists()]
    if still_missing:
        raise StockoutError(
            f"{COMPETITION} downloaded but {', '.join(still_missing)} is not in "
            f"{destination}; the archive layout may have changed."
        )
    return expected
