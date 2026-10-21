#!/usr/bin/env python3
"""Draw the class-count and passage-length charts from the built dataset.

Run scripts/build_classification_set.py first. This script only reads the
saved counts and the labelled file. It does not invent example counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.pipeline import write_plots  # noqa: E402


def main() -> int:
    paths = write_plots(ROOT)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
