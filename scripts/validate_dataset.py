#!/usr/bin/env python3
"""Re-check a derived JSONL file and exit 1 when question leakage is found."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.build_set import read_jsonl  # noqa: E402
from rag_noise.data.validate import validate_pairs  # noqa: E402


def main() -> int:
    path = ROOT / "data" / "processed" / "classification.jsonl"
    report = validate_pairs(read_jsonl(path))
    out = ROOT / "outputs" / "dataset_reports" / "leakage_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "ok", "n_pairs", "n_parent_leaks", "n_question_leaks",
        "n_titles_shared_across_splits", "n_normalized_passages_shared_across_splits",
        "errors",
    )}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
