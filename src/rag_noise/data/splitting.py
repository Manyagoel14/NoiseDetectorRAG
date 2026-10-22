"""Parent-question splits. Variants of one question stay in one split."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from rag_noise.data.labels import LABELS


def _largest_remainder(n: int, ratios: list[float]) -> list[int]:
    if n == 0:
        return [0, 0, 0]
    raw = [n * ratio for ratio in ratios]
    floors = [int(value) for value in raw]
    remainder = n - sum(floors)
    order = sorted(range(3), key=lambda index: raw[index] - floors[index], reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    # Keep a non-empty train split whenever the group is large enough to split.
    if n >= 3 and floors[0] == 0:
        donor = 2 if floors[2] >= floors[1] and floors[2] > 0 else 1
        if floors[donor] > 0:
            floors[donor] -= 1
            floors[0] += 1
    return floors


def assign_splits(
    pairs: list[dict[str, Any]],
    ratios: list[float],
    seed: int,
) -> dict[str, Any]:
    """Assign every pair a split. The grouping key is the normalized question."""
    clusters: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        key = pair["normalized_question"]
        cluster = clusters.setdefault(
            key,
            {"group_id": key, "parent_ids": set(), "sources": set()},
        )
        cluster["parent_ids"].add(pair["parent_id"])
        cluster["sources"].add(pair["source_dataset"])

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters.values():
        stratum = "+".join(sorted(cluster["sources"]))
        by_stratum[stratum].append(cluster)

    rng = random.Random(seed)
    parent_to_split: dict[str, str] = {}
    stratum_sizes: dict[str, dict[str, int]] = {}
    names = ["train", "validation", "test"]
    for stratum, items in sorted(by_stratum.items()):
        items = sorted(items, key=lambda item: item["group_id"])
        rng.shuffle(items)
        counts = _largest_remainder(len(items), ratios)
        stratum_sizes[stratum] = dict(zip(names, counts))
        cursor = 0
        for name, count in zip(names, counts):
            for cluster in items[cursor : cursor + count]:
                for parent_id in cluster["parent_ids"]:
                    parent_to_split[parent_id] = name
            cursor += count

    for pair in pairs:
        pair["split"] = parent_to_split[pair["parent_id"]]

    split_class_counts: dict[str, dict[str, int]] = {
        name: {label: 0 for label in LABELS} for name in names
    }
    split_parent_counts: dict[str, set[str]] = {name: set() for name in names}
    for pair in pairs:
        split_class_counts[pair["split"]][pair["label"]] += 1
        split_parent_counts[pair["split"]].add(pair["parent_id"])

    merged_questions = sum(1 for cluster in clusters.values() if len(cluster["parent_ids"]) > 1)
    return {
        "seed": seed,
        "ratios": ratios,
        "n_question_groups": len(clusters),
        "n_question_groups_covering_multiple_parents": merged_questions,
        "stratum_group_counts": stratum_sizes,
        "parent_counts": {name: len(parents) for name, parents in split_parent_counts.items()},
        "class_counts": split_class_counts,
        "parent_to_split": parent_to_split,
    }
