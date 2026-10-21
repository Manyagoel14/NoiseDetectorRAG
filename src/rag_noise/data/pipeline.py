"""Milestone-1 dataset pipeline: inspect, derive pairs, split, and report."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_noise.data.build_set import (
    build_from_sources,
    cross_class_text_collisions,
    write_jsonl,
)
from rag_noise.data.inspect_raw import inspect_directory, render_inspection_markdown
from rag_noise.data.labels import LABELS
from rag_noise.data.plotting import bar_counts, grouped_split_bars, length_boxes, proportion_bars
from rag_noise.data.splitting import assign_splits
from rag_noise.data.validate import validate_pairs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _imbalance(counts: dict[str, int]) -> dict[str, Any]:
    positive = [value for value in counts.values() if value > 0]
    majority = max(positive) if positive else 0
    minority = min(positive) if positive else 0
    ratios = {}
    for label, value in counts.items():
        ratios[label] = round(majority / value, 4) if value else None
    return {
        "majority_count": majority,
        "minority_count": minority,
        "majority_to_minority": round(majority / minority, 4) if minority else None,
        "majority_to_each_class": ratios,
    }


def _qc_summary(pairs: list[dict[str, Any]], collisions: list[dict[str, Any]]) -> dict[str, Any]:
    relevant_missing = 0
    supportive_with_answer = 0
    prior_mismatch = 0
    prior_total = 0
    ortho_scores: list[float] = []
    illegal_tokens: list[int] = []
    for pair in pairs:
        qc = pair["qc"]
        if pair["label"] == "Relevant" and not qc["answer_in_passage"]:
            relevant_missing += 1
        if pair["label"] == "Supportive Noise" and qc["answer_in_passage"]:
            supportive_with_answer += 1
        if pair["label"] == "Prior Noise":
            prior_total += 1
            if not qc.get("prior_answer_phrase_match"):
                prior_mismatch += 1
        if pair["label"] == "Orthographic Noise":
            ortho_scores.append(qc["similarity_to_gold"])
        if pair["label"] == "Illegal-Sentence Noise":
            illegal_tokens.append(qc["fragment_whitespace_tokens"])
    def _stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.fmean(values), 4),
            "median": statistics.median(values),
        }

    return {
        "relevant_pairs_without_gold_answer_string": relevant_missing,
        "supportive_pairs_that_still_contain_gold_answer": supportive_with_answer,
        "prior_pairs": prior_total,
        "prior_pairs_whose_gold_answer_is_not_the_expected_phrase": prior_mismatch,
        "orthographic_similarity_to_gold": _stats(ortho_scores),
        "illegal_sentence_whitespace_tokens": _stats(illegal_tokens),
        "n_parents_with_identical_text_under_two_labels": len(collisions),
        "identical_text_collisions_preview": collisions[:10],
    }


def _manual_sample(pairs: list[dict[str, Any]], n_hotpot: int, n_prior: int) -> list[dict[str, Any]]:
    hotpot_ids: list[object] = []
    prior_ids: list[object] = []
    for pair in pairs:
        if pair["source_dataset"] == "hotpotqa" and pair["source_example_id"] not in hotpot_ids:
            hotpot_ids.append(pair["source_example_id"])
        if pair["source_dataset"] == "priorqa" and pair["source_example_id"] not in prior_ids:
            prior_ids.append(pair["source_example_id"])
    hotpot_ids = sorted(hotpot_ids)[:n_hotpot]
    prior_ids = sorted(prior_ids)[:n_prior]
    keep_hotpot = set(hotpot_ids)
    keep_prior = set(prior_ids)
    sample = [
        pair
        for pair in pairs
        if (pair["source_dataset"] == "hotpotqa" and pair["source_example_id"] in keep_hotpot)
        or (pair["source_dataset"] == "priorqa" and pair["source_example_id"] in keep_prior)
    ]
    sample.sort(key=lambda pair: (pair["source_dataset"], pair["source_example_id"], pair["label_id"]))
    return sample


def _test_shortfall(class_counts: dict[str, int], minimum: int, labels: list[str]) -> dict[str, int]:
    return {label: class_counts[label] for label in labels if class_counts[label] < minimum}


def report_dir_for(root: Path) -> Path:
    path = root / "outputs" / "dataset_reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_inspection(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Count the raw JSON files. Does not delete or rewrite them."""
    raw_dir = root / config["raw_dirname"]
    report_dir = report_dir_for(root)
    inspection = inspect_directory(raw_dir)
    _write_json(report_dir / "raw_inspection.json", inspection)
    (report_dir / "raw_inspection.md").write_text(render_inspection_markdown(inspection), encoding="utf-8")
    return inspection


def build_classification(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Turn the raw files into labelled pairs, then assign the question split."""
    raw_dir = root / config["raw_dirname"]
    report_dir = report_dir_for(root)
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    commit = config["noiserbench_commit"]
    ratio = str(config["orthographic_ratio"])
    minimum = int(config["min_test_per_class"])
    passage_labels = [label for label in LABELS if label != "Prior Noise"]
    notes: list[str] = []

    sources = list(config["initial_sources"])
    built = build_from_sources(raw_dir, sources, commit, ratio)
    split_info = assign_splits(built["pairs"], config["split_ratios"], int(config["seed"]))
    short_passage = _test_shortfall(split_info["class_counts"]["test"], minimum, passage_labels)
    if short_passage:
        sources = list(config["passage_sources"]) + [config["prior_source"]]
        built = build_from_sources(raw_dir, sources, commit, ratio)
        split_info = assign_splits(built["pairs"], config["split_ratios"], int(config["seed"]))
        notes.append(
            "Initial HotpotQA-plus-PriorQA test split had fewer than "
            f"{minimum} pairs for {short_passage}. Rebuilt from all released "
            "passage-level files plus PriorQA."
        )
    short_final = _test_shortfall(split_info["class_counts"]["test"], minimum, list(LABELS))
    if short_final:
        notes.append(
            f"After the source decision, these classes still have fewer than {minimum} "
            f"test pairs: {short_final}. No rows were synthesized to fill the gap."
        )
    else:
        notes.append(
            f"Every class has at least {minimum} test pairs under the "
            f"{config['split_ratios']} parent-question split."
        )

    collisions = cross_class_text_collisions(built["pairs"])
    qc = _qc_summary(built["pairs"], collisions)
    validation = validate_pairs(built["pairs"])
    sample = _manual_sample(
        built["pairs"],
        int(config["manual_sample_parents"]),
        int(config["manual_sample_prior_parents"]),
    )

    write_jsonl(processed_dir / "classification.jsonl", built["pairs"])
    write_jsonl(report_dir / "manual_sample.jsonl", sample)

    exclusion_path = report_dir / "exclusion_log.csv"
    with exclusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_dataset", "source_example_id", "parent_id", "label", "reason", "detail"],
        )
        writer.writeheader()
        writer.writerows(built["exclusions"])

    _write_json(report_dir / "candidate_lengths.json", built["candidate_lengths"])

    decision = {
        "sources_used": sources,
        "notes": notes,
        "seed": config["seed"],
        "split_ratios": config["split_ratios"],
        "orthographic_ratio": ratio,
        "noiserbench_commit": commit,
        "per_source_records": built["per_source_records"],
        "inventory_counts": built["inventory_counts"],
        "candidate_counts": built["candidate_counts"],
        "cleaned_counts": built["cleaned_counts"],
        "exclusion_reason_counts": built["exclusion_reason_counts"],
        "n_exclusions": len(built["exclusions"]),
        "policy_holdout_counts": built["policy_holdout_counts"],
        "imbalance_inventory": _imbalance(built["inventory_counts"]),
        "imbalance_candidate": _imbalance(built["candidate_counts"]),
        "imbalance_cleaned": _imbalance(built["cleaned_counts"]),
        "split": {
            key: value
            for key, value in split_info.items()
            if key != "parent_to_split"
        },
        "qc": qc,
        "validation_ok": validation["ok"],
        "n_question_leaks": validation["n_question_leaks"],
        "n_parent_leaks": validation["n_parent_leaks"],
        "n_normalized_passages_shared_across_splits": validation[
            "n_normalized_passages_shared_across_splits"
        ],
        "n_titles_shared_across_splits": validation["n_titles_shared_across_splits"],
        "manual_sample_rows": len(sample),
        "manual_sample_hotpot_parents": int(config["manual_sample_parents"]),
        "manual_sample_prior_parents": int(config["manual_sample_prior_parents"]),
    }
    _write_json(report_dir / "dataset_decision.json", decision)
    _write_json(report_dir / "leakage_report.json", validation)
    _write_json(report_dir / "qc_report.json", qc)
    return decision


def write_plots(root: Path) -> list[str]:
    """Draw the class and length charts from the files the build step saved."""
    from rag_noise.data.build_set import read_jsonl

    report_dir = report_dir_for(root)
    plot_dir = root / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    decision_path = report_dir / "dataset_decision.json"
    lengths_path = report_dir / "candidate_lengths.json"
    pairs_path = root / "data" / "processed" / "classification.jsonl"
    for path in (decision_path, lengths_path, pairs_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/build_classification_set.py before making plots."
            )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    candidate_lengths = json.loads(lengths_path.read_text(encoding="utf-8"))
    pairs = read_jsonl(pairs_path)
    cleaned_lengths: dict[str, list[int]] = {label: [] for label in LABELS}
    for pair in pairs:
        cleaned_lengths[pair["label"]].append(len(pair["passage_text"]))

    bar_counts(
        decision["inventory_counts"],
        "Stored NoiserBench variants before any selection",
        "Stored passages",
        plot_dir / "class_counts_inventory.png",
    )
    bar_counts(
        decision["candidate_counts"],
        "One candidate per parent before eligibility filters",
        "Parent questions",
        plot_dir / "class_counts_raw.png",
    )
    bar_counts(
        decision["cleaned_counts"],
        "Cleaned one-pair-per-class dataset",
        "Pairs",
        plot_dir / "class_counts_cleaned.png",
    )
    proportion_bars(
        decision["cleaned_counts"],
        "Class proportions in the cleaned dataset",
        plot_dir / "class_proportions_cleaned.png",
    )
    grouped_split_bars(decision["split"]["class_counts"], plot_dir / "class_counts_by_split.png")
    length_boxes(
        candidate_lengths,
        "Candidate passage length before eligibility filters",
        plot_dir / "length_by_class_raw.png",
    )
    length_boxes(
        cleaned_lengths,
        "Cleaned passage length by class",
        plot_dir / "length_by_class_cleaned.png",
    )
    plot_paths = [
        "outputs/plots/class_counts_inventory.png",
        "outputs/plots/class_counts_raw.png",
        "outputs/plots/class_counts_cleaned.png",
        "outputs/plots/class_proportions_cleaned.png",
        "outputs/plots/class_counts_by_split.png",
        "outputs/plots/length_by_class_raw.png",
        "outputs/plots/length_by_class_cleaned.png",
    ]
    decision["plot_paths"] = plot_paths
    _write_json(report_dir / "dataset_decision.json", decision)
    return plot_paths


def run_pipeline(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    write_inspection(root, config)
    decision = build_classification(root, config)
    write_plots(root)
    return decision
