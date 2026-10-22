"""Leakage and schema checks. Question overlap across splits is a hard failure."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from rag_noise.data.labels import LABEL_TO_ID, LABELS, ORTHOGRAPHIC_RATIO
from rag_noise.data.matching import normalize_for_match

REQUIRED_FIELDS = (
    "example_id",
    "parent_id",
    "source_dataset",
    "source_example_id",
    "question",
    "passage_text",
    "label",
    "label_id",
    "split",
    "provenance",
)


def _passage_hash(text: str) -> str:
    normalized = normalize_for_match(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    parent_splits: dict[str, set[str]] = defaultdict(set)
    question_splits: dict[str, set[str]] = defaultdict(set)
    passage_splits: dict[str, set[str]] = defaultdict(set)
    title_splits: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    label_counts = {label: 0 for label in LABELS}

    for index, pair in enumerate(pairs):
        missing = [field for field in REQUIRED_FIELDS if field not in pair]
        if missing:
            errors.append(f"row {index} missing {missing}")
            continue
        label = pair["label"]
        if label not in LABEL_TO_ID:
            errors.append(f"row {index} has unexpected label {label!r}")
            continue
        label_counts[label] += 1
        if pair["label_id"] != LABEL_TO_ID[label]:
            errors.append(f"row {index} label_id does not match {label}")
        if pair["split"] not in {"train", "validation", "test"}:
            errors.append(f"row {index} has invalid split {pair['split']!r}")
        if pair["example_id"] in seen_ids:
            errors.append(f"duplicate example_id {pair['example_id']}")
        seen_ids.add(pair["example_id"])
        if not isinstance(pair["passage_text"], str) or pair["passage_text"] == "":
            errors.append(f"row {index} has empty passage_text")
        if label == "Orthographic Noise" and str(pair.get("orthographic_ratio")) != ORTHOGRAPHIC_RATIO:
            errors.append(f"row {index} orthographic ratio is not {ORTHOGRAPHIC_RATIO}")
        if label == "Prior Noise" and pair["source_dataset"] != "priorqa":
            errors.append(f"row {index} Prior Noise is not from priorqa")
        if pair["source_dataset"] == "priorqa" and label != "Prior Noise":
            errors.append(f"row {index} priorqa parent emitted {label}")
        parent_splits[pair["parent_id"]].add(pair["split"])
        question_splits[pair["normalized_question"]].add(pair["split"])
        passage_splits[_passage_hash(pair["passage_text"])].add(pair["split"])
        title = pair.get("passage_title") or ""
        if title:
            title_splits[title].add(pair["split"])

    parent_leaks = sorted(parent for parent, splits in parent_splits.items() if len(splits) > 1)
    question_leaks = sorted(
        question for question, splits in question_splits.items() if len(splits) > 1 and question
    )
    if parent_leaks:
        errors.append(f"{len(parent_leaks)} parent ids appear in more than one split")
    if question_leaks:
        errors.append(f"{len(question_leaks)} normalized questions appear in more than one split")

    shared_passages = sum(1 for splits in passage_splits.values() if len(splits) > 1)
    shared_titles = sum(1 for splits in title_splits.values() if len(splits) > 1)
    return {
        "ok": not errors,
        "n_pairs": len(pairs),
        "n_parents": len(parent_splits),
        "n_questions": len(question_leaks) + len([q for q, s in question_splits.items() if len(s) == 1]),
        "label_counts": label_counts,
        "parent_leaks": parent_leaks[:20],
        "n_parent_leaks": len(parent_leaks),
        "question_leaks": question_leaks[:20],
        "n_question_leaks": len(question_leaks),
        "n_normalized_passages_shared_across_splits": shared_passages,
        "n_titles_shared_across_splits": shared_titles,
        "errors": errors,
        "passage_overlap_is_reported_not_failed": True,
    }
