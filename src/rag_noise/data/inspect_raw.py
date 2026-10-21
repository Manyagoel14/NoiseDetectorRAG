"""Raw NoiserBench file inspection. This module does not drop records."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rag_noise.data.labels import FIELD_FOR_LABEL, LABELS, SOURCE_FILES
from rag_noise.data.matching import loose_question_key, normalize_question


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array")
    return data


def _length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
        "stdev": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
    }


def _is_empty_text(value: object) -> bool:
    return not isinstance(value, str) or value.strip() == ""


def _iter_passage_texts(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Yield (field, title, text) for stored passages, without selecting a class."""
    found: list[tuple[str, str, str]] = []
    gold = record.get("gold_text")
    if isinstance(gold, str):
        found.append(("gold_text", "", gold))
    for field in (
        "semantic noise",
        "supportive noise",
        "illegal sentence noise",
    ):
        value = record.get(field)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    title = item.get("title") if isinstance(item.get("title"), str) else ""
                    found.append((field, title, item["text"]))
    datatype = record.get("datatype noise")
    if isinstance(datatype, str):
        found.append(("datatype noise", "", datatype))
    elif isinstance(datatype, list):
        for item in datatype:
            if isinstance(item, str):
                found.append(("datatype noise", "", item))
    for field in ("counterfactual noise", "counterfactual noise2"):
        value = record.get(field)
        if isinstance(value, str):
            found.append((field, "", value))
    orthographic = record.get("orthographic noise")
    if isinstance(orthographic, dict):
        for ratio, text in orthographic.items():
            if isinstance(text, str):
                found.append((f"orthographic noise[{ratio}]", "", text))
    return found


def summarize_records(records: list[dict[str, Any]], source: str, path: Path) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    key_types: dict[str, Counter[str]] = {}
    null_counts: Counter[str] = Counter()
    empty_questions = 0
    empty_gold = 0
    empty_answers = 0
    answer_types: Counter[str] = Counter()
    questions: list[str] = []
    ids: list[object] = []
    char_lengths: list[int] = []
    word_lengths: list[int] = []
    question_chars: list[int] = []
    answer_chars: list[int] = []
    titles: set[str] = set()
    over_word = {256: 0, 384: 0, 512: 0}
    missing_label_field = 0
    split_field_present = 0

    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            key_counts[key] += 1
            key_types.setdefault(key, Counter())[type(value).__name__] += 1
            if value is None:
                null_counts[key] += 1
        if "label" not in record and "noise_label" not in record:
            missing_label_field += 1
        if "split" in record:
            split_field_present += 1
        question = record.get("question")
        if _is_empty_text(question):
            empty_questions += 1
        else:
            questions.append(question)
            question_chars.append(len(question))
        if _is_empty_text(record.get("gold_text")):
            empty_gold += 1
        answers = record.get("gold_answers")
        if answers is None or answers == [] or answers == "":
            empty_answers += 1
        else:
            items = answers if isinstance(answers, list) else [answers]
            for item in items:
                answer_types[type(item).__name__] += 1
                answer_chars.append(len(str(item)))
        if "id" in record:
            ids.append(record["id"])
        for _field, title, text in _iter_passage_texts(record):
            if title:
                titles.add(title)
            words = len(text.split())
            char_lengths.append(len(text))
            word_lengths.append(words)
            for limit in over_word:
                if words > limit:
                    over_word[limit] += 1

    exact_questions = Counter(normalize_question(q) for q in questions)
    duplicate_questions = sum(count - 1 for count in exact_questions.values() if count > 1)
    loose = Counter(loose_question_key(q) for q in questions)
    punctuation_near_duplicates = 0
    for key, count in loose.items():
        if count > 1:
            variants = {q for q in questions if loose_question_key(q) == key}
            normalized = {normalize_question(q) for q in variants}
            if len(normalized) > 1:
                punctuation_near_duplicates += count - 1

    id_info: dict[str, Any] = {"count": len(ids), "unique": len(set(map(str, ids)))}
    numeric_ids = [item for item in ids if isinstance(item, int)]
    if numeric_ids:
        id_info["min"] = min(numeric_ids)
        id_info["max"] = max(numeric_ids)
        id_info["is_contiguous_from_zero"] = sorted(numeric_ids) == list(range(len(numeric_ids)))

    present_fields = sorted(key_counts)
    return {
        "source": source,
        "path": str(path),
        "n_records": len(records),
        "n_unique_questions_normalized": len(exact_questions),
        "n_duplicate_normalized_questions": duplicate_questions,
        "n_punctuation_near_duplicate_questions": punctuation_near_duplicates,
        "n_unique_passage_titles": len(titles),
        "fields": present_fields,
        "field_presence_counts": dict(key_counts),
        "field_types": {key: dict(counter) for key, counter in sorted(key_types.items())},
        "null_counts": dict(null_counts),
        "empty_questions": empty_questions,
        "empty_gold_text": empty_gold,
        "empty_gold_answers": empty_answers,
        "gold_answer_python_types": dict(answer_types),
        "records_without_label_field": missing_label_field,
        "records_with_split_field": split_field_present,
        "id_summary": id_info,
        "question_char_length": _length_stats(question_chars),
        "answer_char_length": _length_stats(answer_chars),
        "stored_passage_char_length": _length_stats(char_lengths),
        "stored_passage_whitespace_tokens": _length_stats(word_lengths),
        "stored_passages_over_whitespace_tokens": over_word,
        "supporting_fact_annotations_present": False,
        "note_on_sequence_length": (
            "Thresholds count whitespace-separated tokens in stored passages. "
            "They are not WordPiece or RoBERTa tokenizer lengths. "
            "NoiserBench does not include HotpotQA supporting-fact lists."
        ),
        "expected_fields_for_eight_class_derivation": {
            label: FIELD_FOR_LABEL[label] for label in LABELS
        },
    }


def inspect_directory(raw_dir: Path) -> dict[str, Any]:
    files = {}
    missing = []
    for source, relative in SOURCE_FILES.items():
        path = raw_dir / relative
        if not path.exists():
            missing.append(str(path))
            continue
        records = load_json_array(path)
        files[source] = summarize_records(records, source, path)
    return {
        "raw_dir": str(raw_dir),
        "missing_files": missing,
        "files": files,
        "total_records": sum(item["n_records"] for item in files.values()),
    }


def render_inspection_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Raw NoiserBench inspection",
        "",
        "Counts below were computed from the local JSON files. Records were not removed.",
        "",
        f"Total records across found files: {report['total_records']}",
        "",
    ]
    if report["missing_files"]:
        lines.append("Missing files:")
        for path in report["missing_files"]:
            lines.append(f"- {path}")
        lines.append("")
    for source, item in report["files"].items():
        words = item["stored_passage_whitespace_tokens"]
        lines.extend(
            [
                f"## {source}",
                "",
                f"- Records: {item['n_records']}",
                f"- Unique normalized questions: {item['n_unique_questions_normalized']}",
                f"- Duplicate normalized questions: {item['n_duplicate_normalized_questions']}",
                f"- Empty questions: {item['empty_questions']}",
                f"- Empty gold text: {item['empty_gold_text']}",
                f"- Empty gold answers: {item['empty_gold_answers']}",
                f"- Records without a label field: {item['records_without_label_field']}",
                f"- Records with a split field: {item['records_with_split_field']}",
                f"- Unique passage titles: {item['n_unique_passage_titles']}",
                f"- Gold-answer Python types: {item['gold_answer_python_types']}",
                f"- Passage whitespace tokens: min {words['min']}, max {words['max']}, "
                f"mean {words['mean']}, median {words['median']}, stdev {words['stdev']}",
                f"- Passages over 256 / 384 / 512 whitespace tokens: "
                f"{item['stored_passages_over_whitespace_tokens']}",
                f"- Identifier summary: {item['id_summary']}",
                "",
            ]
        )
    lines.append(report["files"][next(iter(report["files"]))]["note_on_sequence_length"] if report["files"] else "")
    lines.append("")
    return "\n".join(lines)
