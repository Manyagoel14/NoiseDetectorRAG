#!/usr/bin/env python3
"""Inspect every released NoiserBench JSON file without dropping records."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.config import load_config  # noqa: E402
from rag_noise.data.inspect_raw import inspect_directory, render_inspection_markdown  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs" / "dataset.yaml")
    report = inspect_directory(ROOT / config["raw_dirname"])
    out = ROOT / "outputs" / "dataset_reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_inspection.json").write_text(
        __import__("json").dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    text = render_inspection_markdown(report)
    (out / "raw_inspection.md").write_text(text, encoding="utf-8")
    print(text)
    return 0 if not report["missing_files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
