"""The audit script confirms the known import bug on a tiny fake checkout."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_noiserbench  # noqa: E402


class AuditFixtureTests(unittest.TestCase):
    def test_import_mismatch_is_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "utils").mkdir()
            (repo / "retriever").mkdir()
            (repo / "data_process").mkdir()
            (repo / "README.md").write_text("# NoiserBench\n", encoding="utf-8")
            (repo / "utils" / "config.py").write_text(
                'DATA_PATH = {"nq": "../data/single_hop/nq.json"}\n',
                encoding="utf-8",
            )
            (repo / "utils" / "normalize_answer.py").write_text("def normalize(text):\n    return text\n", encoding="utf-8")
            (repo / "data_process" / "data_loader.py").write_text(
                'def create():\n    names = ["gsm8k"]\n    return item["gold_text"], item["counterfactual_noise"]\n',
                encoding="utf-8",
            )
            (repo / "retriever" / "passage_retrieval.py").write_text(
                "from utils.normalize_answers import is_answer_in_text\n"
                "item['semantic_text'] = []\n"
                "item['supportive_text'] = []\n"
                "while score >= 0.20:\n"
                "    pass\n"
                "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"
                "SentenceTransformer('./all-MiniLM-L6-v2')\n",
                encoding="utf-8",
            )
            findings = {item.issue_id: item.status for item in audit_noiserbench.check_source(repo)}
            self.assertEqual(findings["NB-05"], "CONFIRMED")
            self.assertEqual(findings["NB-03"], "CONFIRMED")
            self.assertEqual(findings["NB-01"], "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
