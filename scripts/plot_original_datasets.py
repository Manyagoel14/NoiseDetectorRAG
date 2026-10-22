#!/usr/bin/env python3
"""Plot every passage stored in the original HotpotQA and PriorQA files.

Run scripts/download_noiserbench.py and scripts/build_classification_set.py
first. This script does not change the raw files or the labelled dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.original_plots import write_original_plots  # noqa: E402


def main() -> int:
    summary = write_original_plots(ROOT)
    print(json.dumps({
        "hotpotqa_stored_counts": summary["hotpotqa_stored_counts"],
        "hotpotqa_kept_counts": summary["hotpotqa_kept_counts"],
        "priorqa_stored_counts": summary["priorqa_stored_counts"],
    }, indent=2))
    for path in summary["plots"]:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
