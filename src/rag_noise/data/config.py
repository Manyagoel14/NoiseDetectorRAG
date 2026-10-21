"""Load the dataset configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rag_noise.data.labels import (
    NOISERBENCH_COMMIT,
    ORTHOGRAPHIC_RATIO,
    PASSAGE_SOURCES,
    PRIOR_SOURCE,
)


def default_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "split_ratios": [0.8, 0.1, 0.1],
        "min_test_per_class": 30,
        "orthographic_ratio": ORTHOGRAPHIC_RATIO,
        "initial_sources": ["hotpotqa", PRIOR_SOURCE],
        "passage_sources": list(PASSAGE_SOURCES),
        "prior_source": PRIOR_SOURCE,
        "noiserbench_commit": NOISERBENCH_COMMIT,
        "raw_dirname": "data/raw/noiserbench",
        "manual_sample_parents": 20,
        "manual_sample_prior_parents": 5,
        "expected_prior_answer": "The question is factually incorrect",
    }


def load_config(path: Path | None) -> dict[str, Any]:
    config = default_config()
    if path is None:
        return config
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    config.update(loaded)
    ratios = config["split_ratios"]
    if len(ratios) != 3:
        raise ValueError("split_ratios must contain train, validation, and test")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1, got {ratios}")
    return config
