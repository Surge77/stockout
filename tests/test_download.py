"""Kaggle acquisition. Everything here is offline except the one marked test."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from stockout.data import download
from stockout.errors import DownloadError


def test_an_existing_extraction_is_not_re_downloaded(tmp_path: Path) -> None:
    """The guard that stops a re-run clobbering a directory someone is working against.

    The autouse network block in conftest means this test would fail loudly if `fetch`
    reached for a socket, which is exactly the assertion being made.
    """
    (tmp_path / download.TRAIN_CSV).write_text("date,store\n", encoding="utf-8")
    assert download.fetch(tmp_path) == tmp_path


def test_the_credential_message_names_both_required_steps() -> None:
    """A valid token is necessary and not sufficient; the rules must be accepted too.

    This is the single most likely first-run failure, and a raw 403 traceback does not
    tell anyone that a web page needs visiting.
    """
    assert "kaggle.json" in download._CREDENTIAL_HELP
    assert "rules" in download._CREDENTIAL_HELP
    assert "403" in download._CREDENTIAL_HELP


def test_a_rejected_download_becomes_a_domain_error_carrying_the_help_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kaggle's 403 is the expected failure, so it must arrive with instructions."""

    class _RefusingApi:
        def competition_download_files(self, *args: object, **kwargs: object) -> None:
            raise OSError("403 Forbidden")

    monkeypatch.setattr(download, "_authenticated_api", lambda: _RefusingApi())
    with pytest.raises(DownloadError, match="rules"):
        download.fetch(tmp_path)


def test_a_missing_archive_after_download_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SilentApi:
        def competition_download_files(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(download, "_authenticated_api", lambda: _SilentApi())
    with pytest.raises(DownloadError, match="Kaggle wrote nothing"):
        download.fetch(tmp_path)


def _archive(target: Path, *, nested: bool) -> None:
    """Write a zip shaped like Kaggle's, optionally with the CSVs zipped a second time."""
    target.mkdir(parents=True, exist_ok=True)
    outer = target / f"{download.KAGGLE_COMPETITION}.zip"
    with zipfile.ZipFile(outer, "w") as bundle:
        if nested:
            inner_path = target / f"{download.TRAIN_CSV}.zip"
            with zipfile.ZipFile(inner_path, "w") as inner:
                inner.writestr(download.TRAIN_CSV, "date,store\n")
            bundle.write(inner_path, f"{download.TRAIN_CSV}.zip")
            inner_path.unlink()
        else:
            bundle.writestr(download.TRAIN_CSV, "date,store\n")
            bundle.writestr(download.STORE_CSV, "store\n")


def test_a_flat_archive_is_unpacked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Api:
        def competition_download_files(self, *args: object, **kwargs: object) -> None:
            _archive(tmp_path, nested=False)

    monkeypatch.setattr(download, "_authenticated_api", lambda: _Api())
    target = download.fetch(tmp_path)
    assert (target / download.TRAIN_CSV).read_text(encoding="utf-8") == "date,store\n"


def test_csvs_zipped_a_second_time_are_also_unpacked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kaggle sometimes nests each CSV in its own zip inside the bundle."""

    class _Api:
        def competition_download_files(self, *args: object, **kwargs: object) -> None:
            _archive(tmp_path, nested=True)

    monkeypatch.setattr(download, "_authenticated_api", lambda: _Api())
    target = download.fetch(tmp_path)
    assert (target / download.TRAIN_CSV).exists()
    assert not list(target.glob("*.csv.zip"))


def test_an_archive_without_the_training_file_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Api:
        def competition_download_files(self, *args: object, **kwargs: object) -> None:
            with zipfile.ZipFile(tmp_path / f"{download.KAGGLE_COMPETITION}.zip", "w") as bundle:
                bundle.writestr("readme.txt", "nothing useful")

    monkeypatch.setattr(download, "_authenticated_api", lambda: _Api())
    with pytest.raises(DownloadError, match="missing after unpacking"):
        download.fetch(tmp_path)


@pytest.mark.integration
def test_fetch_downloads_the_real_archive(tmp_path: Path) -> None:
    """Needs kaggle.json *and* accepted competition rules. Excluded from the default run."""
    target = download.fetch(tmp_path)
    assert (target / download.TRAIN_CSV).exists()
    assert (target / download.STORE_CSV).exists()
