"""Matplotlib figures computed from measured class counts and lengths."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag_noise.data.labels import LABELS


def _ordered(counts: dict[str, int]) -> list[int]:
    missing = [label for label in LABELS if label not in counts]
    if missing:
        raise KeyError(f"Class counts are missing {missing}")
    return [counts[label] for label in LABELS]


def bar_counts(counts: dict[str, int], title: str, ylabel: str, path: Path) -> None:
    values = _ordered(counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(LABELS)), values, color="#3C6E71")
    axis.set_xticks(range(len(LABELS)), list(LABELS), rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(0, max(values + [1]) * 1.15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def proportion_bars(counts: dict[str, int], title: str, path: Path) -> None:
    values = _ordered(counts)
    total = sum(values)
    if total == 0:
        raise ValueError("Cannot plot proportions for an empty dataset")
    shares = [value / total for value in values]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(range(len(LABELS)), shares, color="#284B63")
    axis.set_xticks(range(len(LABELS)), list(LABELS), rotation=30, ha="right")
    axis.set_ylabel("Proportion of cleaned pairs")
    axis.set_title(title)
    axis.set_ylim(0, max(shares) * 1.15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def grouped_split_bars(split_counts: dict[str, dict[str, int]], path: Path) -> None:
    splits = ["train", "validation", "test"]
    for split in splits:
        _ordered(split_counts[split])
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    import numpy as np

    positions = np.arange(len(LABELS))
    width = 0.25
    colors = {"train": "#3C6E71", "validation": "#D9A05B", "test": "#8C3A4B"}
    for offset, split in enumerate(splits):
        values = _ordered(split_counts[split])
        axis.bar(positions + (offset - 1) * width, values, width, label=split, color=colors[split])
    axis.set_xticks(positions, list(LABELS), rotation=30, ha="right")
    axis.set_ylabel("Pairs")
    axis.set_title("Class counts in the train, validation, and test splits")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def length_boxes(lengths: dict[str, list[int]], title: str, path: Path) -> None:
    series = [lengths.get(label, []) for label in LABELS]
    if all(len(item) == 0 for item in series):
        raise ValueError("No lengths were available to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.boxplot(series, tick_labels=list(LABELS), showfliers=False)
    axis.set_ylabel("Passage characters")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def bar_named(names: list[str], values: list[int], title: str, ylabel: str, path: Path) -> None:
    if len(names) != len(values) or not names:
        raise ValueError("Bar chart needs one value for every name")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 0.9), 5))
    axis.bar(range(len(names)), values, color="#3C6E71")
    axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(0, max(values + [1]) * 1.15)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def grouped_count_bars(
    left_name: str,
    left: dict[str, int],
    right_name: str,
    right: dict[str, int],
    title: str,
    path: Path,
    order: list[str] | None = None,
) -> None:
    labels = order or list(left)
    missing = [label for label in labels if label not in left or label not in right]
    if missing:
        raise KeyError(f"Grouped counts are missing {missing}")
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(labels))
    width = 0.38
    axis.bar(positions - width / 2, [left[label] for label in labels], width, label=left_name, color="#8C3A4B")
    axis.bar(positions + width / 2, [right[label] for label in labels], width, label=right_name, color="#3C6E71")
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel("Passages")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def paired_length_boxes(
    left_name: str,
    left: dict[str, list[int]],
    right_name: str,
    right: dict[str, list[int]],
    title: str,
    path: Path,
    order: list[str],
) -> None:
    """Two boxplots per class: every stored passage, and the one passage we kept."""
    import numpy as np

    for label in order:
        if label not in left or label not in right:
            raise KeyError(f"Length comparison is missing {label}")
        if len(left[label]) == 0 or len(right[label]) == 0:
            raise ValueError(f"{label} has no lengths to plot")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(13, 5.5))
    centers = np.arange(len(order)) * 2
    left_plot = axis.boxplot(
        [left[label] for label in order],
        positions=centers - 0.35,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    right_plot = axis.boxplot(
        [right[label] for label in order],
        positions=centers + 0.35,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for box in left_plot["boxes"]:
        box.set_facecolor("#E8A0A8")
    for box in right_plot["boxes"]:
        box.set_facecolor("#8FBFB5")
    axis.set_xticks(centers, order, rotation=30, ha="right")
    axis.set_ylabel("Passage characters")
    axis.set_title(title)
    axis.legend(
        [left_plot["boxes"][0], right_plot["boxes"][0]],
        [left_name, right_name],
        loc="upper right",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
