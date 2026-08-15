"""Chart styling and saving. The charts themselves live in the notebook, by policy."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from stockout.plots import STYLE, apply_style, save


@pytest.fixture
def figure():
    fig, axis = plt.subplots()
    axis.plot([1, 2, 3], [1, 4, 9])
    yield fig
    plt.close(fig)


def test_apply_style_installs_the_house_defaults() -> None:
    apply_style()
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["legend.frameon"] is False


def test_the_style_keys_are_all_real_matplotlib_settings() -> None:
    """A typo in an rc key is silently ignored by matplotlib, so check them explicitly."""
    for key in STYLE:
        assert key in plt.rcParams


def test_save_writes_a_png_and_returns_its_path(figure, tmp_path: Path) -> None:
    path = save(figure, "q1_demand_by_weekday", directory=tmp_path)
    assert path == tmp_path / "q1_demand_by_weekday.png"
    assert path.stat().st_size > 0


def test_save_creates_the_directory_if_it_is_missing(figure, tmp_path: Path) -> None:
    """`reports/` is gitignored, so a fresh clone has no such directory to write into."""
    path = save(figure, "chart", directory=tmp_path / "reports")
    assert path.exists()
