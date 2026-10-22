#!/usr/bin/env python3
"""Run the milestone-1 inspection, derivation, plots, and leakage checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.config import load_config  # noqa: E402
from rag_noise.data.pipeline import run_pipeline  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs" / "dataset.yaml")
    decision = run_pipeline(ROOT, config)
    print(json.dumps({key: decision[key] for key in (
        "sources_used",
        "notes",
        "per_source_records",
        "candidate_counts",
        "cleaned_counts",
        "exclusion_reason_counts",
        "validation_ok",
        "n_question_leaks",
        "n_parent_leaks",
        "n_titles_shared_across_splits",
        "n_normalized_passages_shared_across_splits",
    )}, indent=2))
    return 0 if decision["validation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
