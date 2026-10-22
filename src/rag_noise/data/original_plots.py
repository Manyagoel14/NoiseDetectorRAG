"""Plots of every passage stored in the original HotpotQA and PriorQA files.

The final dataset keeps one passage per class. These plots keep every stored
passage, so the two pictures can be compared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_noise.data.build_set import read_jsonl
from rag_noise.data.inspect_raw import load_json_array
from rag_noise.data.labels import LABELS, SOURCE_FILES
from rag_noise.data.plotting import bar_named, grouped_count_bars, paired_length_boxes

HOTPOT_ORDER = [label for label in LABELS if label != "Prior Noise"]


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def hotpot_stored_lengths(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Character length of every stored HotpotQA passage, including ones we did not keep."""
    lengths = {label: [] for label in HOTPOT_ORDER}
    for record in records:
        gold = record.get("gold_text")
        if _nonempty(gold):
            lengths["Relevant"].append(len(gold))
        for label, field in (
            ("Semantic Noise", "semantic noise"),
            ("Supportive Noise", "supportive noise"),
            ("Illegal-Sentence Noise", "illegal sentence noise"),
        ):
            items = record.get(field)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and _nonempty(item.get("text")):
                        lengths[label].append(len(item["text"]))
        datatype = record.get("datatype noise")
        if isinstance(datatype, str) and _nonempty(datatype):
            lengths["Datatype Noise"].append(len(datatype))
        elif isinstance(datatype, list):
            for item in datatype:
                if _nonempty(item):
                    lengths["Datatype Noise"].append(len(item))
        for field in ("counterfactual noise", "counterfactual noise2"):
            text = record.get(field)
            if _nonempty(text):
                lengths["Counterfactual Noise"].append(len(text))
        orthographic = record.get("orthographic noise")
        if isinstance(orthographic, dict):
            for text in orthographic.values():
                if _nonempty(text):
                    lengths["Orthographic Noise"].append(len(text))
    return lengths


def orthographic_lengths_by_ratio(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    for record in records:
        orthographic = record.get("orthographic noise")
        if not isinstance(orthographic, dict):
            continue
        for ratio, text in orthographic.items():
            if _nonempty(text):
                buckets.setdefault(str(ratio), []).append(len(text))
    return dict(sorted(buckets.items(), key=lambda item: float(item[0])))


def prior_stored_lengths(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Lengths of the PriorQA fields. Only gold_text is used as Prior Noise."""
    lengths = {
        "Prior gold_text (kept)": [],
        "counterfactual noise (not kept)": [],
        "counterfactual noise2 (not kept)": [],
        "background notes (not a class)": [],
    }
    for record in records:
        gold = record.get("gold_text")
        if _nonempty(gold):
            lengths["Prior gold_text (kept)"].append(len(gold))
        for key, label in (
            ("counterfactual noise", "counterfactual noise (not kept)"),
            ("counterfactual noise2", "counterfactual noise2 (not kept)"),
        ):
            text = record.get(key)
            if _nonempty(text):
                lengths[label].append(len(text))
        background = record.get("background")
        if isinstance(background, dict):
            for text in background.values():
                if _nonempty(text):
                    lengths["background notes (not a class)"].append(len(text))
    return lengths


def selected_hotpot_lengths(pairs: list[dict[str, Any]]) -> dict[str, list[int]]:
    lengths = {label: [] for label in HOTPOT_ORDER}
    for pair in pairs:
        if pair["source_dataset"] == "hotpotqa" and pair["label"] in lengths:
            lengths[pair["label"]].append(len(pair["passage_text"]))
    return lengths


def _counts(lengths: dict[str, list[int]]) -> dict[str, int]:
    return {label: len(values) for label, values in lengths.items()}


def write_original_plots(root: Path) -> dict[str, Any]:
    raw_dir = root / "data" / "raw" / "noiserbench"
    plot_dir = root / "outputs" / "plots" / "original"
    report_dir = root / "outputs" / "dataset_reports"
    plot_dir.mkdir(parents=True, exist_ok=True)
    hotpot = load_json_array(raw_dir / SOURCE_FILES["hotpotqa"])
    prior = load_json_array(raw_dir / SOURCE_FILES["priorqa"])
    pairs_path = root / "data" / "processed" / "classification.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/classification.jsonl. "
            "Run scripts/build_classification_set.py first."
        )
    pairs = read_jsonl(pairs_path)

    hotpot_lengths = hotpot_stored_lengths(hotpot)
    selected = selected_hotpot_lengths(pairs)
    prior_lengths = prior_stored_lengths(prior)
    ratio_lengths = orthographic_lengths_by_ratio(hotpot)
    hotpot_counts = _counts(hotpot_lengths)
    selected_counts = _counts(selected)
    prior_counts = _counts(prior_lengths)

    bar_named(
        list(hotpot_counts),
        list(hotpot_counts.values()),
        "Original HotpotQA file: every stored passage",
        "Stored passages",
        plot_dir / "hotpotqa_stored_counts.png",
    )
    grouped_count_bars(
        "Every passage in the original file",
        hotpot_counts,
        "One passage kept in our dataset",
        selected_counts,
        "HotpotQA: original stored passages versus the passages we kept",
        plot_dir / "hotpotqa_stored_vs_kept_counts.png",
        order=HOTPOT_ORDER,
    )
    paired_length_boxes(
        "Every stored passage",
        hotpot_lengths,
        "The one passage we kept",
        selected,
        "HotpotQA passage length: all stored texts versus the one we kept",
        plot_dir / "hotpotqa_stored_vs_kept_lengths.png",
        order=HOTPOT_ORDER,
    )
    bar_named(
        list(ratio_lengths),
        [len(values) for values in ratio_lengths.values()],
        "Original HotpotQA orthographic copies at each damage level",
        "Passages",
        plot_dir / "hotpotqa_orthographic_ratio_counts.png",
    )
    figure_lengths = {f"ratio {ratio}": values for ratio, values in ratio_lengths.items()}
    _named_boxes(
        figure_lengths,
        "Original HotpotQA: length of each orthographic damage level",
        plot_dir / "hotpotqa_orthographic_ratio_lengths.png",
    )
    bar_named(
        list(prior_counts),
        list(prior_counts.values()),
        "Original PriorQA file: fields we kept and fields we left out",
        "Stored texts",
        plot_dir / "priorqa_stored_counts.png",
    )
    _named_boxes(
        prior_lengths,
        "Original PriorQA: length of each stored field",
        plot_dir / "priorqa_field_lengths.png",
    )

    summary = {
        "hotpotqa_records": len(hotpot),
        "priorqa_records": len(prior),
        "hotpotqa_stored_counts": hotpot_counts,
        "hotpotqa_kept_counts": selected_counts,
        "priorqa_stored_counts": prior_counts,
        "orthographic_ratio_counts": {ratio: len(values) for ratio, values in ratio_lengths.items()},
        "plots": [
            "outputs/plots/original/hotpotqa_stored_counts.png",
            "outputs/plots/original/hotpotqa_stored_vs_kept_counts.png",
            "outputs/plots/original/hotpotqa_stored_vs_kept_lengths.png",
            "outputs/plots/original/hotpotqa_orthographic_ratio_counts.png",
            "outputs/plots/original/hotpotqa_orthographic_ratio_lengths.png",
            "outputs/plots/original/priorqa_stored_counts.png",
            "outputs/plots/original/priorqa_field_lengths.png",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "original_vs_kept.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _named_boxes(lengths: dict[str, list[int]], title: str, path: Path) -> None:
    names = list(lengths)
    series = [lengths[name] for name in names]
    if any(len(values) == 0 for values in series):
        raise ValueError(f"A series in {title!r} is empty")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.boxplot(series, tick_labels=names, showfliers=False)
    axis.set_ylabel("Characters")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# Re-export so a reader of this module can find the eight-class helper.
__all__ = [
    "write_original_plots",
    "hotpot_stored_lengths",
    "prior_stored_lengths",
]
