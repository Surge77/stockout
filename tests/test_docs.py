"""Three ways the prose can drift from the code, each of which has already happened here.

Documentation is not usually worth testing. These three are, because each has a machine-
checkable referent and each failed silently for two releases:

- `README.md` opened with three commands that exited 2, because the refactor that deleted
  them never reached the file.
- `src/` and `pyproject.toml` cited ADR 0015, 0017, 0018 and 0019 in eight places, and none
  of those files existed. A code comment pointing at an absent decision record asserts an
  audit trail that is not there, which is worse than citing nothing.
- Links between documents rotted whenever one was renamed.

**History is exempt, and deliberately so.** `CHANGELOG.md` and the superseded ADRs describe
what the package used to do and must go on naming `frontier`, `calibration` and
`models/gbm.py`. A decision log edited to match the present is not a log. Only documents
that read as *current* are checked.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from stockout.cli import _parser

ROOT = Path(__file__).resolve().parents[1]

#: Documents that describe the package as it is now. A reader opening any of these expects
#: every command in it to run today.
CURRENT_DOCS: tuple[str, ...] = (
    "README.md",
    "MODEL_CARD.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "docs/architecture.md",
    "docs/data-dictionary.md",
    "docs/glossary.md",
    "docs/questions.md",
    "docs/results.md",
)

#: Superseded by ADR 0014 and kept as history. They name the modules they were about.
SUPERSEDED_ADRS: frozenset[int] = frozenset(range(7, 14))

_SUBCOMMAND = re.compile(r"stockout ([a-z][a-z-]*)")
_LINK = re.compile(r"\[[^\]]+\]\(([^)#]+?)(?:#[^)]*)?\)")
_ADR_REFERENCE = re.compile(r"ADR (\d{4})")


def _offered_subcommands() -> set[str]:
    for action in _parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("the parser has no subcommands")


def _adr_numbers_on_disk() -> set[int]:
    return {
        int(path.name[:4])
        for path in (ROOT / "docs" / "decisions").glob("0*.md")
    }


@pytest.mark.parametrize("document", CURRENT_DOCS)
def test_every_command_a_current_document_names_actually_exists(document: str) -> None:
    """`python -m stockout <x>` in prose must be an `<x>` the parser accepts.

    The README shipped `backtest --model gbm`, `calibration` and `frontier` for two
    releases after the code behind them was deleted. All three exited 2.
    """
    named = set(_SUBCOMMAND.findall((ROOT / document).read_text(encoding="utf-8")))
    absent = sorted(named - _offered_subcommands())
    assert not absent, f"{document} names subcommands the CLI does not offer: {absent}"


def test_every_adr_cited_from_the_source_exists() -> None:
    """A comment citing ADR 0015 is a promise that `docs/decisions/0015-*.md` is there."""
    on_disk = _adr_numbers_on_disk()
    cited: dict[int, set[str]] = {}
    for source in [*(ROOT / "src").rglob("*.py"), ROOT / "pyproject.toml"]:
        for number in _ADR_REFERENCE.findall(source.read_text(encoding="utf-8")):
            cited.setdefault(int(number), set()).add(str(source.relative_to(ROOT)))

    missing = {number: sorted(where) for number, where in cited.items() if number not in on_disk}
    assert not missing, f"cited from the source, absent from docs/decisions: {missing}"


def test_the_adr_index_lists_every_adr_on_disk() -> None:
    """An ADR nobody links to is one nobody reads."""
    index = (ROOT / "docs" / "decisions" / "README.md").read_text(encoding="utf-8")
    linked = {int(target[:4]) for target in _LINK.findall(index) if target[:4].isdigit()}
    assert _adr_numbers_on_disk() - linked == set()


def test_every_superseded_adr_says_so_at_the_top() -> None:
    """A reader following a link from a 2026 commit message must land somewhere honest."""
    for number in sorted(SUPERSEDED_ADRS):
        (path,) = (ROOT / "docs" / "decisions").glob(f"{number:04d}-*.md")
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:6])
        assert "Superseded by" in head, f"{path.name} describes deleted code without saying so"


def test_every_relative_link_between_documents_resolves() -> None:
    """Including from the history, which may name deleted modules but not dead files."""
    broken = []
    for document in ROOT.rglob("*.md"):
        if ".venv" in document.parts:
            continue
        for target in _LINK.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (document.parent / target).exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not broken, f"broken relative links: {broken}"
