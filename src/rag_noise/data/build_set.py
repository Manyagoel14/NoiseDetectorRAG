"""Derive one question–passage pair per class from NoiserBench fields.

The raw files are not modified. Passages that are not selected are counted in
the inventory. They are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rag_noise.data.inspect_raw import load_json_array
from rag_noise.data.labels import (
    EXPECTED_PRIOR_ANSWER,
    FIELD_FOR_LABEL,
    LABEL_TO_ID,
    LABELS,
    ORTHOGRAPHIC_RATIO,
    SOURCE_FILES,
)
from rag_noise.data.matching import (
    contains_answer,
    nfc_or_none,
    normalize_question,
    serialize_answers,
)


PRESERVE_EXACT = {"Orthographic Noise", "Illegal-Sentence Noise"}


def _example_id(parent_id: str, label: str, field_name: str, variant_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    raw = f"{parent_id}|{label}|{field_name}|{variant_index}|{digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _question_ok(record: dict[str, Any]) -> bool:
    return _nonempty(record.get("question"))


def _exclusion(
    source: str,
    record: dict[str, Any],
    label: str,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "source_dataset": source,
        "source_example_id": record.get("id"),
        "parent_id": f"{source}:{record.get('id')}",
        "label": label,
        "reason": reason,
        "detail": detail,
    }


def _pair(
    source: str,
    record: dict[str, Any],
    label: str,
    text: str,
    title: str,
    variant_index: int,
    selection_rule: str,
    source_file: str,
    commit: str,
    orthographic_ratio: str | None,
    extra_qc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent_id = f"{source}:{record.get('id')}"
    preserve = label in PRESERVE_EXACT
    nfc = nfc_or_none(text, preserve_exact=preserve)
    answers = record.get("gold_answers")
    qc = {
        "answer_in_passage": contains_answer(text, answers),
        "nfc_changed": nfc != text,
    }
    if label == "Prior Noise":
        serialized = serialize_answers(answers)
        qc["prior_answer_phrase_match"] = serialized == [EXPECTED_PRIOR_ANSWER]
    if label == "Orthographic Noise":
        gold = record.get("gold_text") if isinstance(record.get("gold_text"), str) else ""
        # Compare a prefix so the check stays linear in the number of passages.
        qc["similarity_to_gold"] = round(
            SequenceMatcher(None, gold[:500], text[:500]).ratio(), 4
        )
        qc["similarity_prefix_characters"] = 500
    if extra_qc:
        qc.update(extra_qc)
    return {
        "example_id": _example_id(parent_id, label, FIELD_FOR_LABEL[label], variant_index, text),
        "parent_id": parent_id,
        "source_dataset": source,
        "source_example_id": record.get("id"),
        "question": record.get("question"),
        "normalized_question": normalize_question(record["question"]) if _question_ok(record) else "",
        "passage_title": title,
        "passage_text": text,
        "passage_text_nfc": nfc,
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "gold_answers": serialize_answers(answers),
        "field_name": FIELD_FOR_LABEL[label],
        "variant_index": variant_index,
        "orthographic_ratio": orthographic_ratio,
        "split": None,
        "provenance": {
            "noiserbench_commit": commit,
            "source_file": source_file,
            "label_source": "noiserbench_field",
            "selection_rule": selection_rule,
        },
        "qc": qc,
    }


def _first_dict_text(items: object) -> tuple[int, str, str] | None:
    if not isinstance(items, list):
        return None
    for index, item in enumerate(items):
        if isinstance(item, dict) and _nonempty(item.get("text")):
            title = item.get("title") if isinstance(item.get("title"), str) else ""
            return index, title, item["text"]
    return None


def _inventory_for_record(record: dict[str, Any], source: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if _nonempty(record.get("gold_text")) and source != "priorqa":
        counts["Relevant"] += 1
    if source == "priorqa" and _nonempty(record.get("gold_text")):
        counts["Prior Noise"] += 1
    semantic = record.get("semantic noise")
    if isinstance(semantic, list):
        counts["Semantic Noise"] += sum(
            1 for item in semantic if isinstance(item, dict) and _nonempty(item.get("text"))
        )
    supportive = record.get("supportive noise")
    if isinstance(supportive, list):
        counts["Supportive Noise"] += sum(
            1 for item in supportive if isinstance(item, dict) and _nonempty(item.get("text"))
        )
    illegal = record.get("illegal sentence noise")
    if isinstance(illegal, list):
        counts["Illegal-Sentence Noise"] += sum(
            1 for item in illegal if isinstance(item, dict) and _nonempty(item.get("text"))
        )
    datatype = record.get("datatype noise")
    if isinstance(datatype, str) and _nonempty(datatype):
        counts["Datatype Noise"] += 1
    elif isinstance(datatype, list):
        counts["Datatype Noise"] += sum(1 for item in datatype if _nonempty(item))
    if _nonempty(record.get("counterfactual noise")) and source != "priorqa":
        counts["Counterfactual Noise"] += 1
    if _nonempty(record.get("counterfactual noise2")) and source != "priorqa":
        counts["Counterfactual Noise"] += 1
    orthographic = record.get("orthographic noise")
    if isinstance(orthographic, dict):
        counts["Orthographic Noise"] += sum(1 for text in orthographic.values() if _nonempty(text))
    return counts


def _candidate_length(record: dict[str, Any], source: str, ratio: str) -> dict[str, int]:
    """Character length of the one passage we would consider before eligibility filters."""
    lengths: dict[str, int] = {}
    if source == "priorqa":
        if _nonempty(record.get("gold_text")):
            lengths["Prior Noise"] = len(record["gold_text"])
        return lengths
    if _nonempty(record.get("gold_text")):
        lengths["Relevant"] = len(record["gold_text"])
    picked = _first_dict_text(record.get("semantic noise"))
    if picked:
        lengths["Semantic Noise"] = len(picked[2])
    picked = _first_dict_text(record.get("supportive noise"))
    if picked:
        lengths["Supportive Noise"] = len(picked[2])
    picked = _first_dict_text(record.get("illegal sentence noise"))
    if picked:
        lengths["Illegal-Sentence Noise"] = len(picked[2])
    datatype = record.get("datatype noise")
    if isinstance(datatype, str) and _nonempty(datatype):
        lengths["Datatype Noise"] = len(datatype)
    elif isinstance(datatype, list):
        for item in datatype:
            if _nonempty(item):
                lengths["Datatype Noise"] = len(item)
                break
    if _nonempty(record.get("counterfactual noise")):
        lengths["Counterfactual Noise"] = len(record["counterfactual noise"])
    orthographic = record.get("orthographic noise")
    if isinstance(orthographic, dict) and _nonempty(orthographic.get(ratio)):
        lengths["Orthographic Noise"] = len(orthographic[ratio])
    return lengths


def select_pairs(
    source: str,
    record: dict[str, Any],
    source_file: str,
    commit: str,
    orthographic_ratio: str = ORTHOGRAPHIC_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "id" not in record:
        return [], [
            {
                "source_dataset": source,
                "source_example_id": None,
                "parent_id": None,
                "label": "",
                "reason": "missing_id",
                "detail": "Record has no id",
            }
        ]

    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    question_ok = _question_ok(record)

    if source == "priorqa":
        text = record.get("gold_text")
        if not question_ok:
            exclusions.append(_exclusion(source, record, "Prior Noise", "empty_question", "Question is empty"))
        elif not _nonempty(text):
            exclusions.append(_exclusion(source, record, "Prior Noise", "empty_passage", "gold_text is empty"))
        else:
            pairs.append(
                _pair(
                    source,
                    record,
                    "Prior Noise",
                    text,
                    "",
                    0,
                    "priorqa gold_text paired with the false-premise question; "
                    "counterfactual fields from PriorQA are not emitted",
                    source_file,
                    commit,
                    None,
                )
            )
        return pairs, exclusions

    def require_question(label: str) -> bool:
        if question_ok:
            return True
        exclusions.append(_exclusion(source, record, label, "empty_question", "Question is empty"))
        return False

    gold = record.get("gold_text")
    if "gold_text" not in record:
        exclusions.append(_exclusion(source, record, "Relevant", "missing_field", "gold_text absent"))
    elif not _nonempty(gold):
        exclusions.append(_exclusion(source, record, "Relevant", "empty_passage", "gold_text is empty"))
    elif require_question("Relevant"):
        pairs.append(
            _pair(source, record, "Relevant", gold, "", 0, "gold_text", source_file, commit, None)
        )

    semantic = _first_dict_text(record.get("semantic noise"))
    if not isinstance(record.get("semantic noise"), list) or semantic is None:
        exclusions.append(
            _exclusion(source, record, "Semantic Noise", "missing_or_empty_field", "semantic noise has no text")
        )
    elif require_question("Semantic Noise"):
        index, title, text = semantic
        pairs.append(
            _pair(
                source,
                record,
                "Semantic Noise",
                text,
                title,
                index,
                "semantic noise, first non-empty passage",
                source_file,
                commit,
                None,
            )
        )

    datatype_text = None
    datatype_index = 0
    datatype = record.get("datatype noise")
    if isinstance(datatype, str):
        datatype_text = datatype if _nonempty(datatype) else None
    elif isinstance(datatype, list):
        for index, item in enumerate(datatype):
            if _nonempty(item):
                datatype_text = item
                datatype_index = index
                break
    if datatype_text is None:
        exclusions.append(
            _exclusion(source, record, "Datatype Noise", "missing_or_empty_field", "datatype noise has no text")
        )
    elif require_question("Datatype Noise"):
        pairs.append(
            _pair(
                source,
                record,
                "Datatype Noise",
                datatype_text,
                "",
                datatype_index,
                "datatype noise, first non-empty string",
                source_file,
                commit,
                None,
            )
        )

    illegal = _first_dict_text(record.get("illegal sentence noise"))
    if illegal is None:
        exclusions.append(
            _exclusion(
                source,
                record,
                "Illegal-Sentence Noise",
                "missing_or_empty_field",
                "illegal sentence noise has no text",
            )
        )
    elif require_question("Illegal-Sentence Noise"):
        index, title, text = illegal
        pairs.append(
            _pair(
                source,
                record,
                "Illegal-Sentence Noise",
                text,
                title,
                index,
                "illegal sentence noise, first non-empty fragment",
                source_file,
                commit,
                None,
                {"fragment_whitespace_tokens": len(text.split())},
            )
        )

    counterfactual = record.get("counterfactual noise")
    if not _nonempty(counterfactual):
        exclusions.append(
            _exclusion(
                source,
                record,
                "Counterfactual Noise",
                "missing_or_empty_field",
                "counterfactual noise is empty",
            )
        )
    elif require_question("Counterfactual Noise"):
        pairs.append(
            _pair(
                source,
                record,
                "Counterfactual Noise",
                counterfactual,
                "",
                0,
                "counterfactual noise; counterfactual noise2 is held out",
                source_file,
                commit,
                None,
            )
        )

    supportive = record.get("supportive noise")
    chosen_supportive = None
    answer_hits = 0
    if isinstance(supportive, list):
        for index, item in enumerate(supportive):
            if not isinstance(item, dict) or not _nonempty(item.get("text")):
                continue
            if contains_answer(item["text"], record.get("gold_answers")):
                answer_hits += 1
                continue
            title = item.get("title") if isinstance(item.get("title"), str) else ""
            chosen_supportive = (index, title, item["text"])
            break
    if chosen_supportive is None:
        reason = "supportive_answer_present_in_all" if answer_hits else "missing_or_empty_field"
        exclusions.append(
            _exclusion(
                source,
                record,
                "Supportive Noise",
                reason,
                f"No supportive passage without a gold answer; answer hits={answer_hits}",
            )
        )
    elif require_question("Supportive Noise"):
        index, title, text = chosen_supportive
        pairs.append(
            _pair(
                source,
                record,
                "Supportive Noise",
                text,
                title,
                index,
                "supportive noise, first passage whose normalized text does not contain a gold answer",
                source_file,
                commit,
                None,
                {"skipped_supportive_passages_containing_answer": answer_hits},
            )
        )

    orthographic = record.get("orthographic noise")
    ortho_text = orthographic.get(orthographic_ratio) if isinstance(orthographic, dict) else None
    if not _nonempty(ortho_text):
        exclusions.append(
            _exclusion(
                source,
                record,
                "Orthographic Noise",
                "missing_or_empty_field",
                f"orthographic noise[{orthographic_ratio}] is missing or empty",
            )
        )
    elif ortho_text == record.get("gold_text"):
        exclusions.append(
            _exclusion(
                source,
                record,
                "Orthographic Noise",
                "orthographic_identical_to_gold",
                f"Ratio {orthographic_ratio} did not change gold_text",
            )
        )
    elif require_question("Orthographic Noise"):
        pairs.append(
            _pair(
                source,
                record,
                "Orthographic Noise",
                ortho_text,
                "",
                0,
                f"orthographic noise ratio {orthographic_ratio}; other ratios are held out",
                source_file,
                commit,
                orthographic_ratio,
            )
        )
    return pairs, exclusions


def build_from_sources(
    raw_dir: Path,
    sources: list[str],
    commit: str,
    orthographic_ratio: str = ORTHOGRAPHIC_RATIO,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    inventory: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    candidate_lengths: dict[str, list[int]] = {label: [] for label in LABELS}
    per_source_records: dict[str, int] = {}
    holdout: Counter[str] = Counter()

    for source in sources:
        relative = SOURCE_FILES[source]
        path = raw_dir / relative
        records = load_json_array(path)
        per_source_records[source] = len(records)
        for record in records:
            inventory.update(_inventory_for_record(record, source))
            _count_holdouts(record, source, orthographic_ratio, holdout)
            for label, length in _candidate_length(record, source, orthographic_ratio).items():
                candidate_counts[label] += 1
                candidate_lengths[label].append(length)
            built, skipped = select_pairs(source, record, relative, commit, orthographic_ratio)
            pairs.extend(built)
            exclusions.extend(skipped)

    cleaned_counts = Counter(pair["label"] for pair in pairs)
    return {
        "pairs": pairs,
        "exclusions": exclusions,
        "inventory_counts": {label: inventory.get(label, 0) for label in LABELS},
        "candidate_counts": {label: candidate_counts.get(label, 0) for label in LABELS},
        "cleaned_counts": {label: cleaned_counts.get(label, 0) for label in LABELS},
        "candidate_lengths": candidate_lengths,
        "exclusion_reason_counts": dict(Counter(item["reason"] for item in exclusions)),
        "policy_holdout_counts": dict(holdout),
        "per_source_records": per_source_records,
        "sources": list(sources),
    }


def _count_holdouts(record: dict[str, Any], source: str, ratio: str, holdout: Counter[str]) -> None:
    if source == "priorqa":
        if _nonempty(record.get("counterfactual noise")):
            holdout["priorqa_counterfactual_noise"] += 1
        if _nonempty(record.get("counterfactual noise2")):
            holdout["priorqa_counterfactual_noise2"] += 1
        return
    if _nonempty(record.get("counterfactual noise2")):
        holdout["counterfactual_noise2"] += 1
    orthographic = record.get("orthographic noise")
    if isinstance(orthographic, dict):
        for key, text in orthographic.items():
            if key != ratio and _nonempty(text):
                holdout[f"orthographic_ratio_{key}"] += 1


def cross_class_text_collisions(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_parent: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for pair in pairs:
        by_parent[pair["parent_id"]][pair["passage_text"]].add(pair["label"])
    collisions = []
    for parent_id, texts in by_parent.items():
        for text, labels in texts.items():
            if len(labels) > 1:
                collisions.append(
                    {
                        "parent_id": parent_id,
                        "labels": sorted(labels),
                        "passage_preview": text[:180],
                    }
                )
    return collisions


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
