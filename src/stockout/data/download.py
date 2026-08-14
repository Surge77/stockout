"""Fetch the Rossmann archive from Kaggle.

The import style here is deliberate. `import kaggle` authenticates as a side effect of
the import itself and raises `OSError` at module load when no credential exists — which
would make `stockout describe --sample ...` fail on a machine that never intends to
touch Kaggle. Importing the API class directly defers authentication to `authenticate()`,
where a failure can be explained.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

from ..config import KAGGLE_COMPETITION, RAW_DIR
from ..errors import DownloadError

logger = logging.getLogger(__name__)

TRAIN_CSV = "train.csv"
STORE_CSV = "store.csv"

_RULES_URL = f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules"

_CREDENTIAL_HELP = (
    "Kaggle rejected the request. Two separate things are required and a valid token "
    "alone is not enough:\n"
    "  1. An API token at ~/.kaggle/kaggle.json "
    "(kaggle.com -> Settings -> API -> Create New Token).\n"
    f"  2. Accepting the competition rules, signed in, at {_RULES_URL}\n"
    "Until step 2 is done the API answers 403 for a token that is otherwise fine."
)


def fetch(destination: Path | None = None, *, force: bool = False) -> Path:
    """Download and unpack the competition archive. Returns the directory written.

    Refuses to overwrite an existing extraction unless `force` is set — re-downloading
    is slow and silently clobbering a directory someone has been working against is
    worse than an error message.
    """
    target = destination or RAW_DIR
    target.mkdir(parents=True, exist_ok=True)

    train_path = target / TRAIN_CSV
    if train_path.exists() and not force:
        logger.info("%s already present; pass force=True to re-download", train_path)
        return target

    api = _authenticated_api()
    try:
        api.competition_download_files(KAGGLE_COMPETITION, path=str(target), quiet=True)
    except Exception as exc:  # kaggle raises bare ApiException / OSError variants
        raise DownloadError(f"{_CREDENTIAL_HELP}\n\nunderlying error: {exc}") from exc

    archive = target / f"{KAGGLE_COMPETITION}.zip"
    if not archive.exists():
        raise DownloadError(f"expected {archive} after download; Kaggle wrote nothing")

    _unpack(archive, target)
    if not train_path.exists():
        raise DownloadError(f"{TRAIN_CSV} missing after unpacking {archive}")
    return target


def _authenticated_api() -> Any:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - kaggle is a hard dependency
        raise DownloadError("the kaggle package is not installed") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise DownloadError(f"{_CREDENTIAL_HELP}\n\nunderlying error: {exc}") from exc
    return api


def _unpack(archive: Path, target: Path) -> None:
    """Extract the outer archive, then any CSV zips Kaggle nests inside it."""
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)

    for nested in target.glob("*.csv.zip"):
        with zipfile.ZipFile(nested) as inner:
            inner.extractall(target)
        nested.unlink()
