"""Unit tests for derivation, splits, leakage checks, and plots."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.build_set import select_pairs  # noqa: E402
from rag_noise.data.labels import NOISERBENCH_COMMIT  # noqa: E402
from rag_noise.data.matching import contains_answer  # noqa: E402
from rag_noise.data.plotting import bar_counts  # noqa: E402
from rag_noise.data.splitting import assign_splits  # noqa: E402
from rag_noise.data.validate import validate_pairs  # noqa: E402


def hotpot_record() -> dict:
    return {
        "id": 1,
        "question": "Who won?",
        "gold_answers": ["Ada"],
        "gold_text": "Ada won the prize.",
        "semantic noise": [{"id": 1, "title": "Boats", "text": "A ferry crossed the harbour."}],
        "supportive noise": [
            {"id": 1, "title": "Prize", "text": "Ada won the prize in 1999."},
            {"id": 2, "title": "City", "text": "The final was played in Paris."},
        ],
        "datatype noise": ["Ada won the prize. See https://example.com\nprint('x')"],
        "illegal sentence noise": [{"id": 1, "text": "riors ip ious ml"}],
        "counterfactual noise": "Bob won the prize according to a fake report.",
        "counterfactual noise2": "This second false passage stays out of the derived set.",
        "orthographic noise": {"0.2": "Ada wn the pize.", "0.8": "zzzz"},
    }


class MatchingTests(unittest.TestCase):
    def test_boolean_answer_uses_whole_words(self) -> None:
        self.assertTrue(contains_answer("The answer is no.", [False]))
        self.assertFalse(contains_answer("The value is known.", [False]))

    def test_answer_match_does_not_rewrite_the_passage(self) -> None:
        passage = "The pize was wn yesterday."
        self.assertFalse(contains_answer(passage, ["Ada"]))
        self.assertEqual(passage, "The pize was wn yesterday.")


class BuildTests(unittest.TestCase):
    def test_one_pair_per_passage_class_and_orthography_is_preserved(self) -> None:
        pairs, exclusions = select_pairs("hotpotqa", hotpot_record(), "hotpotqa.json", NOISERBENCH_COMMIT)
        labels = {pair["label"] for pair in pairs}
        self.assertEqual(
            labels,
            {
                "Relevant",
                "Semantic Noise",
                "Datatype Noise",
                "Illegal-Sentence Noise",
                "Counterfactual Noise",
                "Supportive Noise",
                "Orthographic Noise",
            },
        )
        supportive = next(pair for pair in pairs if pair["label"] == "Supportive Noise")
        self.assertEqual(supportive["variant_index"], 1)
        self.assertEqual(supportive["passage_text"], "The final was played in Paris.")
        orthographic = next(pair for pair in pairs if pair["label"] == "Orthographic Noise")
        self.assertEqual(orthographic["passage_text"], "Ada wn the pize.")
        self.assertEqual(orthographic["orthographic_ratio"], "0.2")
        self.assertFalse(any(item["label"] == "Prior Noise" for item in exclusions))

    def test_priorqa_emits_only_prior(self) -> None:
        record = {
            "id": 3,
            "question": "What did Leonardo invent for the atomic bomb?",
            "gold_answers": ["The question is factually incorrect"],
            "gold_text": "Leonardo da Vinci made no contribution to the atomic bomb.",
            "counterfactual noise": "A false passage that must not become its own class here.",
        }
        pairs, _exclusions = select_pairs("priorqa", record, "priorqa.json", NOISERBENCH_COMMIT)
        self.assertEqual([pair["label"] for pair in pairs], ["Prior Noise"])
        self.assertIn("Leonardo", pairs[0]["passage_text"])

    def test_empty_gold_text_is_excluded_and_logged(self) -> None:
        record = hotpot_record()
        record["gold_text"] = "  "
        _pairs, exclusions = select_pairs("hotpotqa", record, "hotpotqa.json", NOISERBENCH_COMMIT)
        reasons = [item["reason"] for item in exclusions if item["label"] == "Relevant"]
        self.assertEqual(reasons, ["empty_passage"])


class SplitTests(unittest.TestCase):
    def test_duplicate_questions_stay_together(self) -> None:
        pairs = []
        for index in range(20):
            question = f"Question {index}?"
            pairs.append(_pair(f"hotpotqa:{index}", "hotpotqa", question))
        pairs.append(_pair("nq:99", "nq", "Question 0?"))
        summary = assign_splits(pairs, [0.8, 0.1, 0.1], seed=42)
        splits = {pair["parent_id"]: pair["split"] for pair in pairs}
        self.assertEqual(splits["hotpotqa:0"], splits["nq:99"])
        self.assertEqual(summary["n_question_groups_covering_multiple_parents"], 1)

    def test_validator_rejects_question_leakage(self) -> None:
        first = _pair("hotpotqa:1", "hotpotqa", "Same question?")
        second = _pair("hotpotqa:1", "hotpotqa", "Same question?")
        first["split"] = "train"
        second["split"] = "test"
        second["example_id"] = "different"
        second["label"] = "Semantic Noise"
        second["label_id"] = 1
        report = validate_pairs([first, second])
        self.assertFalse(report["ok"])
        self.assertGreater(report["n_parent_leaks"], 0)


class PlotTests(unittest.TestCase):
    def test_bar_chart_is_written_from_the_supplied_counts(self) -> None:
        counts = {
            "Relevant": 4,
            "Semantic Noise": 4,
            "Datatype Noise": 3,
            "Illegal-Sentence Noise": 4,
            "Counterfactual Noise": 4,
            "Supportive Noise": 2,
            "Orthographic Noise": 4,
            "Prior Noise": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.png"
            bar_counts(counts, "Test counts", "Pairs", path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


def _pair(parent_id: str, source: str, question: str) -> dict:
    return {
        "example_id": parent_id,
        "parent_id": parent_id,
        "source_dataset": source,
        "source_example_id": parent_id.split(":")[1],
        "question": question,
        "normalized_question": question.lower().strip(),
        "passage_title": "",
        "passage_text": "A passage.",
        "label": "Relevant",
        "label_id": 0,
        "split": None,
        "orthographic_ratio": None,
        "provenance": {},
        "qc": {},
    }


if __name__ == "__main__":
    unittest.main()
