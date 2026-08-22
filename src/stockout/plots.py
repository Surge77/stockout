"""Matplotlib and seaborn charts for the notebook. Nothing here is called by the CLI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

FIGURE_SIZE = (9, 6)


def correlation_heatmap(frame: pd.DataFrame, *, title: str = "Correlation") -> Figure:
    """Every column against every other. All columns here are numeric, so this is total."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    sns.heatmap(frame.corr(), annot=True, fmt=".2f", cmap="viridis", ax=axes)
    axes.set_title(title)
    figure.tight_layout()
    return figure


def predicted_against_actual(
    y_true: pd.Series, y_pred: pd.Series, *, title: str = "Predicted against actual"
) -> Figure:
    """The regression diagnostic: points on the diagonal are exact."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    sns.scatterplot(x=y_true, y=y_pred, s=12, alpha=0.4, ax=axes)
    limits = [min(y_true.min(), min(y_pred)), max(y_true.max(), max(y_pred))]
    axes.plot(limits, limits, color="crimson", linewidth=1)
    axes.set_xlabel("actual")
    axes.set_ylabel("predicted")
    axes.set_title(title)
    figure.tight_layout()
    return figure


def scores_bar(results: pd.DataFrame, *, metric: str, title: str = "") -> Figure:
    """The comparison table as a chart, keeping the table's simplest-first order."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    sns.barplot(data=results, x="model", y=metric, ax=axes, hue="model", legend=False)
    axes.set_title(title or f"{metric} by model")
    axes.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure


def save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path
