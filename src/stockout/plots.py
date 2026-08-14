"""Figure styling and saving. No chart builders live here.

The split is deliberate and matches the rest of the repo: anything with one correct
answer is in the package and tested; the charts themselves are analytical choices and
belong in the notebook, next to the reasoning that produced them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import REPORTS_DIR

#: Matplotlib rc overrides. Muted, legible at the size GitHub renders a PNG in a README,
#: and free of the default blue that says "nobody touched this".
STYLE: dict[str, Any] = {
    "figure.figsize": (9, 5),
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.frameon": False,
}


def apply_style() -> None:
    """Install the house style into the active matplotlib session."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)


def save(fig: Any, name: str, *, directory: Path | None = None) -> Path:
    """Write `fig` to `reports/<name>.png` and return the path.

    `reports/` is gitignored: every chart must be reproducible from the notebook, so
    committing one would only create a stale artefact nobody notices is stale.
    """
    target = (directory or REPORTS_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.png"
    fig.savefig(path)
    return path
