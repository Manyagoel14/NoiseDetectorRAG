#!/usr/bin/env python3
"""Download the eight NoiserBench JSON files pinned to the audited commit."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.labels import NOISERBENCH_COMMIT, RAW_BASE_URL, SOURCE_FILES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    raw_dir = ROOT / "data" / "raw" / "noiserbench"
    manifest = {"noiserbench_commit": NOISERBENCH_COMMIT, "files": []}
    for source, relative in SOURCE_FILES.items():
        destination = raw_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size == 0:
            url = f"{RAW_BASE_URL}/{relative}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, destination)
        else:
            print(f"Reusing {destination}")
        manifest["files"].append(
            {
                "source": source,
                "path": f"data/raw/noiserbench/{relative}",
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    out = ROOT / "data" / "manifests" / "noiserbench_checksums.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
