"""Matplotlib plots from supplied arrays/tables, without fitting or data loading.

All functions return a Figure, save a 300-DPI PNG by default and accept show,
close, dpi and data_source options. close=True removes the figure from pyplot's
registry after saving/showing; the returned object remains inspectable. Use
output_path=None to skip saving and close=False for further interactive edits.
This module respects the caller's chosen matplotlib backend.
"""

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._validation import as_binary_labels
from .config import FIGURES_DIR


def _values(values, name, ndim):
    """Require nonempty, finite real numeric plotting values."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numeric values.")
    if raw.ndim != ndim or raw.size == 0 or 0 in raw.shape:
        raise ValueError(f"{name} must be a nonempty {ndim}D array.")
    return raw.astype(np.float64, copy=True)


def _metric_table(results, columns):
    """Validate a table and bounded metric values, returning a copy."""
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("results must be a nonempty pandas DataFrame.")
    if not results.columns.is_unique or not set(columns).issubset(results.columns):
        raise ValueError(f"results must have unique columns including {columns}.")
    table = results.copy()
    for name in columns:
        if name in ("k", "method"):
            continue
        values = _values(table[name], name, 1)
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} values must be in [0, 1].")
        table[name] = values
    return table


def _style_axis(ax):
    """Apply local styling without changing global matplotlib settings."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def _finish(fig, output_path, show, close, dpi, title, data_source):
    """Save/show a labelled figure and reliably close it when requested."""
    try:
        if isinstance(dpi, bool) or not isinstance(dpi, Integral) or dpi <= 0:
            raise ValueError("dpi must be a positive integer.")
        if not isinstance(data_source, str) or not data_source.strip():
            raise ValueError("data_source must be a nonempty string.")
        fig.suptitle(title, fontsize=16, fontweight="bold", color="#172b4d", y=0.98)
        source = f"Data: {data_source}"
        if data_source.lower() == "synthetic":
            source += " | Development verification only"
        fig.text(0.5, 0.015, source, ha="center", fontsize=9, color="#526278")
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
        if output_path is not None:
            path = Path(output_path)
            if path.suffix.lower() != ".png":
                raise ValueError("output_path must have a .png extension.")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=int(dpi), facecolor="white")
        if show:
            plt.show()
        return fig
    finally:
        if close:
            plt.close(fig)


def plot_explained_variance(
    explained_variance_ratio, *, output_path=FIGURES_DIR / "explained_variance.png",
    show=False, close=True, dpi=300, data_source="unspecified",
):
    """Plot retained PC variance ratios as bars and their cumulative sum."""
    ratios = _values(explained_variance_ratio, "explained_variance_ratio", 1)
    if np.any(ratios < 0) or ratios.sum() > 1 + 1e-8:
        raise ValueError("Variance ratios must be nonnegative with sum <= 1.")
    indices = np.arange(1, len(ratios) + 1)
    cumulative = np.minimum(np.cumsum(ratios), 1)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    _style_axis(ax)
    ax.bar(indices, ratios, color="#3579b8", width=0.65, label="Individual PC")
    ax.plot(indices, cumulative, color="#c56b24", marker="o", linewidth=2, label="Cumulative")
    ax.annotate(f"{cumulative[-1]:.1%}", (indices[-1], cumulative[-1]),
                xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10)
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio", ylim=(0, 1.12))
    ax.set_xticks(indices[::max(1, int(np.ceil(len(indices) / 20)))])
    ax.legend(loc="upper left", frameon=False)
    return _finish(fig, output_path, show, close, dpi, "PCA Explained Variance", data_source)


def plot_pca_loadings(
    components, feature_names, *, output_path=FIGURES_DIR / "pca_loadings.png",
    show=False, close=True, dpi=300, data_source="unspecified",
):
    """Plot signed PCA axis coefficients (rows=PCs, columns=original features).

    These are components_ coefficients, without multiplying by sqrt(variance).
    Supply only the PC rows you want to show. Color limits are symmetric.
    """
    matrix = _values(components, "components", 2)
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature_names must match the component columns.")
    try:
        names = list(feature_names)
    except TypeError as exc:
        raise ValueError("feature_names must match the component columns.") from exc
    if len(names) != matrix.shape[1] or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must match the component columns with nonempty names.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    fig, ax = plt.subplots(figsize=(max(9, min(20, 0.65 * len(names) + 2)), max(4.8, min(16, matrix.shape[0] * 0.5 + 2))))
    limit = float(np.max(np.abs(matrix))) or 1.0
    heatmap = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), labels=[f"PC{i + 1}" for i in range(matrix.shape[0])])
    ax.set(xlabel="Original feature", ylabel="Principal component")
    fig.colorbar(heatmap, ax=ax, pad=0.02, label="Axis coefficient")
    return _finish(fig, output_path, show, close, dpi, "PCA Loading Coefficients", data_source)


def plot_pc1_pc2(
    scores, y, *, output_path=FIGURES_DIR / "pc1_pc2.png",
    show=False, close=True, dpi=300, data_source="unspecified",
):
    """Scatter the first two supplied PCs, colored/shaped by binary labels."""
    matrix = _values(scores, "scores", 2)
    if matrix.shape[1] < 2:
        raise ValueError("scores must contain at least two principal components.")
    labels = as_binary_labels(y, "y", len(matrix))
    fig, ax = plt.subplots(figsize=(9, 5.8))
    _style_axis(ax)
    for value, color, marker in ((0, "#3579b8", "o"), (1, "#c56b24", "^")):
        mask = labels == value
        if mask.any():
            ax.scatter(matrix[mask, 0], matrix[mask, 1], c=color, marker=marker,
                       s=25 if value == 0 else 40, alpha=0.65, edgecolors="none",
                       label=f"Class {value} (n={int(mask.sum())})")
    ax.set(xlabel="PC1", ylabel="PC2")
    ax.legend(frameon=False)
    return _finish(fig, output_path, show, close, dpi, "PC1-PC2 Class Distribution", data_source)


def plot_performance_vs_k(
    results, *, output_path=FIGURES_DIR / "performance_vs_k.png",
    show=False, close=True, dpi=300, data_source="unspecified",
):
    """Plot F1/PR-AUC and optional ROC-AUC against sorted, unique positive k."""
    metrics = ["f1", "pr_auc"]
    if isinstance(results, pd.DataFrame) and "roc_auc" in results.columns:
        metrics.append("roc_auc")
    table = _metric_table(results, ["k", *metrics])
    k = _values(table["k"], "k", 1)
    if table["k"].dtype.kind == "b" or np.any(k <= 0) or np.any(k != np.floor(k)) or len(np.unique(k)) != len(k):
        raise ValueError("k must contain unique positive integers.")
    table = table.sort_values("k")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    _style_axis(ax)
    for name, label, color, marker in zip(metrics, ["F1", "PR-AUC", "ROC-AUC"],
                                          ["#3579b8", "#c56b24", "#248777"], ["o", "s", "^"]):
        ax.plot(table.k, table[name], label=label, color=color, marker=marker, linewidth=2)
    ax.set(xlabel="Number of principal components (k)", ylabel="Score", ylim=(0, 1.05))
    sorted_k = table.k.to_numpy()
    ax.set_xticks(sorted_k[::max(1, int(np.ceil(len(sorted_k) / 20)))])
    ax.legend(frameon=False)
    return _finish(fig, output_path, show, close, dpi, "Classification Performance vs k", data_source)


def plot_method_comparison(
    results, *, output_path=FIGURES_DIR / "method_comparison.png",
    show=False, close=True, dpi=300, data_source="unspecified",
):
    """Plot grouped F1/PR-AUC/ROC-AUC bars in the supplied method row order."""
    table = _metric_table(results, ["method", "f1", "pr_auc", "roc_auc"])
    names = table.method.tolist()
    if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("method names must be unique nonempty strings.")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    _style_axis(ax)
    indices = np.arange(len(table))
    for offset, name, label, color in zip([-1, 0, 1], ["f1", "pr_auc", "roc_auc"],
                                         ["F1", "PR-AUC", "ROC-AUC"], ["#3579b8", "#c56b24", "#248777"]):
        bars = ax.bar(indices + offset * 0.24, table[name], width=0.24, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xticks(indices, labels=names)
    ax.set(xlabel="Feature representation", ylabel="Score", ylim=(0, 1.15))
    ax.legend(loc="upper center", ncols=3, frameon=False)
    return _finish(fig, output_path, show, close, dpi, "Original vs PCA vs SVD", data_source)

