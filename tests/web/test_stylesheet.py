"""One stylesheet, no framework — and therefore one flat namespace to collide in.

Every page loads `style.css` whole, so a selector defined twice is not two rules on two
pages; it is the later one winning everywhere. There is no build step to warn about it and
no component boundary to contain it.

That is not hypothetical. `.bar` is the site header. The shelf table's progress bars were
added with the same class name twenty lines from the bottom of the same file, and being
later at equal specificity they won: every header on every page became an 8px rounded
green sliver with its own nav spilling out across the `<h1>` beneath it. The file already
carried a comment saying `.bar` was taken — a comment is not a check.

These tests are that check. They read the stylesheet as text rather than parsing CSS,
which is enough for the failure that actually happened: the same top-level class defined
in two places.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

STYLESHEET = Path(__file__).resolve().parents[2] / "src" / "stockout_web" / "static" / "style.css"

#: A rule that starts a line with a single class and opens a block: `.name {` or
#: `.name,`. Deliberately narrow — descendant and compound selectors (`.bar nav`,
#: `.tag.tag-admin`) are refinements of a class, not redefinitions of it.
_TOP_LEVEL_CLASS = re.compile(r"^(\.[a-z][a-z0-9-]*)\s*\{", re.MULTILINE)


def _definitions() -> Counter[str]:
    return Counter(_TOP_LEVEL_CLASS.findall(STYLESHEET.read_text(encoding="utf-8")))


def test_no_class_is_defined_twice() -> None:
    """The whole file is one namespace, so a repeated name is the later rule winning."""
    repeated = {name: count for name, count in _definitions().items() if count > 1}

    assert repeated == {}, f"defined more than once: {repeated}"


def test_the_site_header_keeps_its_own_rule() -> None:
    """`.bar` is the header. Pinned by name because this is the one that broke."""
    assert _definitions()[".bar"] == 1


def test_the_shelf_bars_do_not_reuse_the_header_name() -> None:
    css = STYLESHEET.read_text(encoding="utf-8")

    assert ".shelf-bar" in css
