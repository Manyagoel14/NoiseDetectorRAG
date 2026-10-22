#!/usr/bin/env python3
"""Build the labelled question-passage file and the train/validation/test split.

Run scripts/download_noiserbench.py first. This script reads the raw JSON files
and writes the derived dataset. It does not draw the charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.config import load_config  # noqa: E402
from rag_noise.data.pipeline import build_classification  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs" / "dataset.yaml")
    decision = build_classification(ROOT, config)
    print(json.dumps({
        "sources_used": decision["sources_used"],
        "cleaned_counts": decision["cleaned_counts"],
        "n_exclusions": decision["n_exclusions"],
        "validation_ok": decision["validation_ok"],
        "n_question_leaks": decision["n_question_leaks"],
        "n_parent_leaks": decision["n_parent_leaks"],
    }, indent=2))
    print("Wrote data/processed/classification.jsonl")
    print("Wrote outputs/dataset_reports/dataset_decision.json")
    print("Wrote outputs/dataset_reports/exclusion_log.csv")
    print("Wrote outputs/dataset_reports/qc_report.json")
    print("Wrote outputs/dataset_reports/leakage_report.json")
    print("Wrote outputs/dataset_reports/manual_sample.jsonl")
    return 0 if decision["validation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
