"""Randomness is allowed in this package, but only where it is the point.

The previous version of this file banned scikit-learn outright: no `train_test_split`,
no `random_state`, no import. That made the leakage problem impossible to demonstrate,
which is a strange way to teach it — the project could assert that a shuffled split was
wrong and could never show it.

So the guard was not deleted, it was **scoped**. Three rules, each checked by parsing the
AST rather than grepping the text, because a text search cannot tell code from the
docstrings explaining the code and would therefore be defeated by deleting the
explanation.

1. **Shuffled splitters live in exactly one module.** `split/strategies.py` exists to run
   `train_test_split` against a time-ordered split and measure the difference. Anywhere
   else, a shuffled splitter is the bug ADR 0003 is about.
2. **`np.random` lives in the generator.** `data/synth.py` takes an explicit seed and
   produces a byte-identical frame from it. Nothing else should be inventing numbers.
3. **A seed, once used, is never `None`.** Reproducibility is not the same thing as
   having no randomness. `random_state=None` means two runs of the same command produce
   two different tables, and the second one silently wins.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "stockout"

#: Splitters that shuffle. Every one of them is legitimate in the module whose job is to
#: demonstrate what shuffling costs, and nowhere else.
SHUFFLING_SPLITTERS = frozenset(
    {
        "train_test_split",
        "ShuffleSplit",
        "StratifiedShuffleSplit",
        "KFold",
        "StratifiedKFold",
    }
)

#: The one module allowed to name them, relative to `src/stockout`.
SPLITTER_EXEMPTION = Path("split") / "strategies.py"

#: The one module allowed to draw random numbers directly.
RANDOMNESS_EXEMPTION = "synth.py"


def _python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_there_are_source_files_to_check() -> None:
    """A guard that runs over nothing passes over nothing."""
    assert len(_python_files()) >= 15


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_shuffled_splitters_appear_only_where_they_are_the_subject(path: Path) -> None:
    if path.relative_to(SRC) == SPLITTER_EXEMPTION:
        pytest.skip("split/strategies.py exists to demonstrate exactly this")

    offences = [
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(_tree(path))
        if (isinstance(node, ast.Name) and node.id in SHUFFLING_SPLITTERS)
        or (isinstance(node, ast.Attribute) and node.attr in SHUFFLING_SPLITTERS)
    ]
    assert not offences, (
        f"{path.name} names a shuffled splitter {sorted(set(offences))}. "
        f"Those belong in {SPLITTER_EXEMPTION.as_posix()}, which measures what they cost."
    )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_a_seed_is_never_left_to_chance(path: Path) -> None:
    """`random_state=None` is not reproducible, and an unreproducible table is a rumour."""
    offences = [
        node.lineno
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.keyword)
        and node.arg == "random_state"
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    assert not offences, f"{path.name}:{offences} passes random_state=None"


def test_the_only_module_drawing_random_numbers_is_the_generator() -> None:
    users = sorted(
        {
            path.name
            for path in _python_files()
            for node in ast.walk(_tree(path))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in {"np", "numpy"}
            and node.attr == "random"
        }
    )
    assert users in ([], [RANDOMNESS_EXEMPTION]), (
        f"{users} draw random numbers directly; only {RANDOMNESS_EXEMPTION} may."
    )
