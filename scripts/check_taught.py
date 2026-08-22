"""Fail if `src/` imports a scikit-learn name that is not taught or not a listed exception.

This replaces a `grep`, which could not see a parenthesised multi-line import: with
`from sklearn.ensemble import (` on its own line the pattern matched an empty tail and the
check passed while hiding every name inside the brackets. Parsing the AST cannot be fooled
by formatting.

    python scripts/check_taught.py          # exit 0 when clean
    python scripts/check_taught.py --strict # also fail on the three documented exceptions
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

#: Every scikit-learn name used in the code cells of the notebooks under
#: `ML/Resource Files` and `ML/Personal Files`, plus `Task_Regression.pdf`.
TAUGHT: frozenset[str] = frozenset(
    {
        "train_test_split",
        "Pipeline",
        "StandardScaler",
        "PolynomialFeatures",
        "LabelEncoder",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "BaggingClassifier",
        "KNeighborsClassifier",
        "SVC",
        "TfidfVectorizer",
        "r2_score",
        "mean_squared_error",
        "accuracy_score",
    }
)

#: Untaught on purpose, added in 0.8.0 by request. Every one carries `taught=False` in the
#: registry and is named in README. A fourth entry here needs the same paperwork.
EXCEPTIONS: frozenset[str] = frozenset(
    {"RandomForestRegressor", "GradientBoostingRegressor", "VotingRegressor"}
)


def sklearn_imports(path: Path) -> list[tuple[int, str]]:
    """Every name imported from a sklearn module in one file, with its line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
            found.extend((alias.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend(
                (alias.lineno, alias.name)
                for alias in node.names
                if alias.name.startswith("sklearn")
            )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="reject the three documented exceptions too",
    )
    args = parser.parse_args(argv)

    allowed = TAUGHT if args.strict else TAUGHT | EXCEPTIONS

    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, name in sklearn_imports(path):
            if name not in allowed:
                label = "exception" if name in EXCEPTIONS else "NOT TAUGHT"
                relative = path.relative_to(SRC.parent)
                violations.append(f"{relative}:{lineno}: {name}  [{label}]")

    if violations:
        print("\n".join(violations))
        print(f"\n{len(violations)} disallowed scikit-learn import(s).")
        return 1

    scope = "taught set only" if args.strict else "taught set + 3 documented exceptions"
    print(f"clean: every scikit-learn import in src/ is within the {scope}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
