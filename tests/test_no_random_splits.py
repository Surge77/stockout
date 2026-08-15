"""The package must contain no randomised splitting, and this proves it structurally.

An earlier version of this check was a `grep` in CI. It failed immediately — on the
docstrings that explain *why* there is no shuffled splitter, and on `__pycache__` files.
A text search cannot tell code from prose about code, which makes it both noisy and,
worse, defeatable by anyone who deletes the explanation.

Parsing the AST looks only at what executes. Docstrings and comments are string and
comment nodes and are never visited; `.pyc` files are never opened. So the assertion is
exactly "no shuffling happens here", not "nobody mentions shuffling".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "stockout"

#: Names that would mean a random split had crept in. `random_state` is included even
#: though it is often harmless elsewhere: nothing in this package should have a seeded
#: source of randomness except the synthetic generator, which takes an explicit `seed`.
BANNED_NAMES = frozenset(
    {
        "train_test_split",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "KFold",
        "StratifiedKFold",
        "shuffle",
        "random_state",
    }
)

BANNED_MODULES = frozenset({"sklearn"})


def _python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_there_are_source_files_to_check() -> None:
    """Guards the guard: an empty file list would make everything below pass vacuously."""
    assert len(_python_files()) >= 15


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_module_performs_a_randomised_split(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            offences.append(f"name {node.id!r} at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_NAMES:
            offences.append(f"attribute {node.attr!r} at line {node.lineno}")
        elif isinstance(node, ast.keyword) and node.arg in BANNED_NAMES:
            offences.append(f"keyword argument {node.arg!r} at line {node.lineno}")

    assert not offences, f"{path.name} performs a randomised split: {'; '.join(offences)}"


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_no_module_imports_scikit_learn(path: Path) -> None:
    """sklearn is not a dependency, and its splitters are the specific hazard."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (roots & BANNED_MODULES), f"{path.name}:{node.lineno} imports scikit-learn"


def test_the_only_module_using_randomness_is_the_generator() -> None:
    """`np.random` may execute in exactly one module, and it takes an explicit seed.

    Also parsed rather than searched, for the same reason as above: a docstring is
    allowed to discuss randomness, and only a call site counts.
    """
    users = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "random"
                and isinstance(node.value, ast.Name)
                and node.value.id in {"np", "numpy", "random"}
            ):
                users.append(path.name)
                break
    assert users == ["synth.py"]
