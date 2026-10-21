"""Fixed label order for the eight-class question–passage task."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "Relevant",
    "Semantic Noise",
    "Datatype Noise",
    "Illegal-Sentence Noise",
    "Counterfactual Noise",
    "Supportive Noise",
    "Orthographic Noise",
    "Prior Noise",
)

LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(LABELS)}

PASSAGE_SOURCES: tuple[str, ...] = (
    "nq",
    "rgb",
    "hotpotqa",
    "2wikimqa",
    "bamboogle",
    "strategyqa",
    "tempqa",
)

PRIOR_SOURCE = "priorqa"

# Paths are relative to the raw NoiserBench data directory.
SOURCE_FILES: dict[str, str] = {
    "nq": "single-hop/nq.json",
    "rgb": "single-hop/rgb.json",
    "hotpotqa": "multi-hop/explicit/hotpotqa.json",
    "2wikimqa": "multi-hop/explicit/2wikimqa.json",
    "bamboogle": "multi-hop/explicit/bamboogle.json",
    "strategyqa": "multi-hop/implicit/strategyqa.json",
    "tempqa": "multi-hop/implicit/tempqa.json",
    "priorqa": "mix-hop/priorqa.json",
}

NOISERBENCH_COMMIT = "447bee38d91996615ecc85c4da4a5b204f6c9837"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/jinyangwu/NoiserBench/"
    + NOISERBENCH_COMMIT
    + "/data"
)

FIELD_FOR_LABEL: dict[str, str] = {
    "Relevant": "gold_text",
    "Semantic Noise": "semantic noise",
    "Datatype Noise": "datatype noise",
    "Illegal-Sentence Noise": "illegal sentence noise",
    "Counterfactual Noise": "counterfactual noise",
    "Supportive Noise": "supportive noise",
    "Orthographic Noise": "orthographic noise",
    "Prior Noise": "gold_text",
}

ORTHOGRAPHIC_RATIO = "0.2"
EXPECTED_PRIOR_ANSWER = "The question is factually incorrect"
