#!/usr/bin/env python3
"""Verify the NoiserBench issues recorded for this project.

The script reads a local NoiserBench checkout and the raw JSON files. It does
not modify either tree. Write-up for the course report:

    python scripts/audit_noiserbench.py \\
        --repo /path/to/NoiserBench \\
        --data data/raw/noiserbench
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_noise.data.inspect_raw import load_json_array  # noqa: E402
from rag_noise.data.labels import NOISERBENCH_COMMIT, SOURCE_FILES  # noqa: E402


@dataclass
class Finding:
    issue_id: str
    title: str
    status: str
    evidence: list[str] = field(default_factory=list)
    impact: str = ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _line(path: Path, needle: str) -> str:
    if not path.exists():
        return f"{path} is absent"
    for number, line in enumerate(_read(path).splitlines(), start=1):
        if needle in line:
            return f"{path.name}:{number}: {line.strip()}"
    return f"{needle!r} was not found in {path}"


def _has(path: Path, needle: str) -> bool:
    return path.exists() and needle in _read(path)


def check_source(repo: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not repo.exists():
        findings.append(
            Finding(
                "NB-00",
                "NoiserBench checkout was not available",
                "UNVERIFIED",
                [f"Expected a checkout at {repo}"],
                "Source-code claims could not be checked on this machine.",
            )
        )
        return findings

    requirement_names = [
        "requirements.txt",
        "pyproject.toml",
        "environment.yml",
        "setup.py",
    ]
    present = [name for name in requirement_names if (repo / name).exists()]
    findings.append(
        Finding(
            "NB-01",
            "The repository has no Python dependency manifest",
            "CONFIRMED" if not present else "NOT_REPRODUCED",
            [f"Looked for {requirement_names}. Found: {present or 'none'}."],
            "A fresh environment cannot be recreated from the repository alone.",
        )
    )

    readme = repo / "README.md"
    readme_text = _read(readme) if readme.exists() else ""
    has_usage = any(token in readme_text.lower() for token in ("pip ", "python ", "usage", "install"))
    findings.append(
        Finding(
            "NB-02",
            "README links the paper and dataset but does not document how to run the code",
            "CONFIRMED" if readme.exists() and not has_usage else "NOT_REPRODUCED",
            [
                "README.md contains the dataset and paper links."
                if "NoiserBench" in readme_text
                else "README.md did not contain the expected project name.",
                f"Usage-like tokens present: {has_usage}.",
            ],
            "Reproducing an experiment requires reading the Python files directly.",
        )
    )

    config = repo / "utils" / "config.py"
    config_text = _read(config) if config.exists() else ""
    findings.append(
        Finding(
            "NB-03",
            "DATA_PATH lists only Natural Questions and points at a directory that is not in the tree",
            "CONFIRMED"
            if "single_hop" in config_text and "single-hop" not in config_text
            else "NOT_REPRODUCED",
            [_line(config, "nq"), f"Config file text: {config_text.strip()}"],
            "The loader cannot find the released files without an external path edit. The files live under data/single-hop, with a hyphen.",
        )
    )

    loader = repo / "data_process" / "data_loader.py"
    loader_text = _read(loader) if loader.exists() else ""
    named = [
        name
        for name in (
            "nq",
            "rgb",
            "hotpotqa",
            "2wikimqa",
            "bamboogle",
            "tempqa",
            "priorqa",
            "strategyqa",
            "gsm8k",
            "gsm8k_hard",
        )
        if name in loader_text
    ]
    data_files = sorted(str(path.relative_to(repo)) for path in (repo / "data").glob("**/*.json")) if (repo / "data").exists() else []
    findings.append(
        Finding(
            "NB-04",
            "The loader names datasets that have no entry in DATA_PATH, including GSM8K",
            "CONFIRMED" if "gsm8k" in loader_text and "gsm8k" not in config_text else "NOT_REPRODUCED",
            [
                _line(loader, "gsm8k"),
                f"Dataset names mentioned in the loader: {named}.",
                f"JSON files visible in this checkout's data/ directory: {data_files or 'none'}.",
                "A sparse checkout can omit data/. The release that we downloaded separately has eight JSON files and no gsm8k file.",
            ],
            "GSM8K is referenced in code and is not one of the eight released NoiserBench JSON files.",
        )
    )

    retrieval = repo / "retriever" / "passage_retrieval.py"
    answer_module = repo / "utils" / "normalize_answer.py"
    wrong_import = repo / "utils" / "normalize_answers.py"
    findings.append(
        Finding(
            "NB-05",
            "Supportive-noise retrieval imports utils.normalize_answers, but the file is normalize_answer.py",
            "CONFIRMED"
            if _has(retrieval, "normalize_answers") and answer_module.exists() and not wrong_import.exists()
            else "NOT_REPRODUCED",
            [
                _line(retrieval, "normalize_answers"),
                f"utils/normalize_answer.py exists: {answer_module.exists()}.",
                f"utils/normalize_answers.py exists: {wrong_import.exists()}.",
            ],
            "The supportive and semantic retrieval script raises ImportError as committed, before it can build passages.",
        )
    )

    findings.append(
        Finding(
            "NB-06",
            "The retrieval script stores semantic_text and supportive_text, while the released JSON uses different field names",
            "CONFIRMED"
            if _has(retrieval, "semantic_text") and _has(retrieval, "supportive_text")
            else "NOT_REPRODUCED",
            [
                _line(retrieval, "semantic_text"),
                _line(retrieval, "supportive_text"),
            ],
            "Even after the import is repaired, this script's output names do not match the released schema. It cannot be treated as a drop-in reader of the JSON files.",
        )
    )

    findings.append(
        Finding(
            "NB-07",
            "Semantic filtering uses a cosine threshold of 0.20 that the main paper text does not state",
            "CONFIRMED" if _has(retrieval, "0.20") else "NOT_REPRODUCED",
            [_line(retrieval, "0.20")],
            "The threshold is an implementation detail of this file. The paper names all-MiniLM-L6-v2 and Contriever but, in the sections we read, does not give 0.20.",
        )
    )

    textnoisr_hits = []
    for path in repo.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".txt"} and "docs/" not in str(path):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "textnoisr" in text.lower():
                textnoisr_hits.append(str(path.relative_to(repo)))
    findings.append(
        Finding(
            "NB-08",
            "textnoisr is not vendored, even though the paper uses it for orthographic noise",
            "CONFIRMED" if not textnoisr_hits else "NOT_REPRODUCED",
            [f"Files mentioning textnoisr: {textnoisr_hits or 'none'}."],
            "Orthographic corruption cannot be rerun from this repository. The released ratios are data, not a generator.",
        )
    )

    generator_needles = ("illegal sentence", "datatype noise", "ChatGPT", "textnoisr")
    generator_hits = {needle: _has(loader, needle) or _has(retrieval, needle) for needle in generator_needles}
    findings.append(
        Finding(
            "NB-09",
            "No generator is present for datatype, illegal-sentence, counterfactual, orthographic, or prior noise",
            "CONFIRMED" if not any(generator_hits.values()) else "NOT_REPRODUCED",
            [
                "Searched data_loader.py and passage_retrieval.py for generator clues.",
                str(generator_hits),
                "passage_retrieval.py builds only supportive and semantic candidates, and it does not run as committed.",
            ],
            "Those five classes can be reused from the released JSON. They cannot be regenerated from this code without writing a new generator and marking it as ours.",
        )
    )

    findings.append(
        Finding(
            "NB-10",
            "The evaluation loader reads golden text and one counterfactual passage, not the other noise fields",
            "CONFIRMED"
            if _has(loader, "counterfactual_noise") and "semantic noise" not in loader_text
            else "NOT_REPRODUCED",
            [
                _line(loader, "gold_text"),
                _line(loader, "counterfactual_noise"),
            ],
            "The shipped loader is a multiple-choice prompt builder. It is not an eight-class dataset builder.",
        )
    )

    split_dirs = [name for name in ("train", "validation", "test", "dev") if (repo / "data" / name).exists()]
    findings.append(
        Finding(
            "NB-11",
            "The data directory has no train, validation, or test split",
            "CONFIRMED" if not split_dirs else "NOT_REPRODUCED",
            [f"Split directories under data/: {split_dirs or 'none'}."],
            "Any detector split has to be created by us. The paper instead samples up to 500 test cases per dataset.",
        )
    )

    pal_imports = []
    for path in repo.rglob("*.py"):
        if "pal/" in str(path.relative_to(repo)).replace("\\", "/") and not str(path.relative_to(repo)).startswith("pal/scripts"):
            continue
        relative = str(path.relative_to(repo))
        if relative.startswith("pal/"):
            continue
        text = _read(path)
        if "import pal" in text or "from pal" in text:
            pal_imports.append(relative)
    findings.append(
        Finding(
            "NB-12",
            "The vendored PAL project is not imported by the noise-benchmark code",
            "CONFIRMED" if not pal_imports else "NOT_REPRODUCED",
            [f"Imports of pal outside pal/: {pal_imports or 'none'}."],
            "PAL is extra code in the tree. It is not part of the seven-noise implementation.",
        )
    )

    findings.append(
        Finding(
            "NB-13",
            "Retrieval code hard-codes CUDA devices and a local MiniLM directory",
            "CONFIRMED" if _has(retrieval, "CUDA_VISIBLE_DEVICES") and _has(retrieval, "all-MiniLM-L6-v2") else "NOT_REPRODUCED",
            [
                _line(retrieval, "CUDA_VISIBLE_DEVICES"),
                _line(retrieval, "all-MiniLM-L6-v2"),
            ],
            "Running the original retriever needs a GPU, a local embedding model, a Wikipedia passage file, and a Contriever index. None of those artifacts are in the repository.",
        )
    )
    return findings


def _status_from_bool(condition: bool) -> str:
    return "CONFIRMED" if condition else "NOT_REPRODUCED"


def check_data(raw_dir: Path) -> list[Finding]:
    if not raw_dir.exists():
        return [
            Finding(
                "NB-14",
                "Released JSON files were not on disk",
                "UNVERIFIED",
                [f"Expected files under {raw_dir}"],
                "Schema claims stay unverified until the JSON files are downloaded.",
            )
        ]

    loaded: dict[str, list] = {}
    missing = []
    for source, relative in SOURCE_FILES.items():
        path = raw_dir / relative
        if not path.exists():
            missing.append(relative)
            continue
        loaded[source] = load_json_array(path)

    findings = [
        Finding(
            "NB-14",
            "All eight released JSON files can be loaded as arrays",
            _status_from_bool(not missing and len(loaded) == 8),
            [
                f"Loaded {len(loaded)} files.",
                f"Missing: {missing or 'none'}.",
                "Record counts: " + ", ".join(f"{name}={len(rows)}" for name, rows in loaded.items()),
            ],
            "These counts are the benchmark release, not the official full HotpotQA training set of 90,447.",
        )
    ]

    def keys_of(source: str) -> set[str]:
        rows = loaded.get(source, [])
        if not rows:
            return set()
        return set(rows[0])

    no_label = all("label" not in row and "noise_label" not in row for rows in loaded.values() for row in rows[:20])
    findings.append(
        Finding(
            "NB-15",
            "Released records do not carry an eight-class label",
            _status_from_bool(bool(loaded) and no_label),
            [
                "Checked the first 20 records of each loaded file for label and noise_label.",
                "HotpotQA keys: " + ", ".join(sorted(keys_of("hotpotqa"))),
            ],
            "A classification row is created only when our code chooses one field and assigns the class. That label is derived, not original.",
        )
    )

    hotpot_keys = keys_of("hotpotqa")
    multi = {
        "semantic noise",
        "supportive noise",
        "datatype noise",
        "illegal sentence noise",
        "counterfactual noise",
        "orthographic noise",
        "gold_text",
    }
    findings.append(
        Finding(
            "NB-16",
            "Each HotpotQA record stores several noise fields at once",
            _status_from_bool(multi.issubset(hotpot_keys)),
            [f"Required fields present: {sorted(multi & hotpot_keys)}."],
            "Exploding every stored passage would make list-valued classes about ten times larger than Relevant. The derived dataset keeps one passage per class.",
        )
    )

    prior_keys = keys_of("priorqa")
    prior_missing = sorted(multi - {"gold_text", "counterfactual noise"} - prior_keys)
    findings.append(
        Finding(
            "NB-17",
            "PriorQA does not contain the passage-noise fields used by the other files",
            _status_from_bool("background" in prior_keys and "semantic noise" not in prior_keys),
            [
                f"PriorQA keys: {sorted(prior_keys)}.",
                f"Passage-noise fields absent from the first record: {prior_missing}.",
            ],
            "Prior noise is a false premise in the question. The corrective text is gold_text. PriorQA counterfactual passages are a second noise type on the same question, so the derived set does not emit them.",
        )
    )

    strategy = loaded.get("strategyqa", [])
    answer_types = Counter(type(item).__name__ for row in strategy for item in row.get("gold_answers", []))
    missing_second = sum(1 for row in strategy if "counterfactual answer2" not in row)
    findings.append(
        Finding(
            "NB-18",
            "StrategyQA answers are booleans, and the second counterfactual field is absent",
            _status_from_bool(answer_types.get("bool", 0) > 0 and missing_second == len(strategy) and strategy),
            [
                f"gold_answers element types: {dict(answer_types)}.",
                f"Records without counterfactual answer2: {missing_second} of {len(strategy)}.",
            ],
            "Answer-containment checks must treat true and false as words. They must not assume every file has two counterfactual strings.",
        )
    )

    ratio_sets = Counter()
    for row in loaded.get("hotpotqa", []):
        value = row.get("orthographic noise")
        if isinstance(value, dict):
            ratio_sets[tuple(value.keys())] += 1
    findings.append(
        Finding(
            "NB-19",
            "Orthographic noise is stored at ten corruption ratios, not as one class",
            _status_from_bool(any("0.2" in keyset and "0.8" in keyset for keyset in ratio_sets)),
            [f"HotpotQA orthographic key sets: { {str(key): count for key, count in ratio_sets.items()} }."],
            "The derived Orthographic class uses ratio 0.2. Ratio 0.01 is almost clean, and 0.8 is close to illegible text.",
        )
    )

    contiguous = {}
    for source, rows in loaded.items():
        ids = [row.get("id") for row in rows]
        contiguous[source] = ids == list(range(len(rows)))
    findings.append(
        Finding(
            "NB-20",
            "Record ids are local integers starting at 0, not official HotpotQA ids",
            _status_from_bool(contiguous.get("hotpotqa") is True),
            [f"ids == 0..n-1 for each file: {contiguous}."],
            "The release cannot be aligned to the official HotpotQA train, validation, and test ids without matching question strings.",
        )
    )

    paper_example = "history transform cover managed that hand black"
    illegal_equal = 0
    illegal_total = 0
    alpha_tokens = 0
    all_tokens = 0
    for row in loaded.get("hotpotqa", []):
        items = row.get("illegal sentence noise")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            illegal_total += 1
            text = item["text"]
            if text.strip() == paper_example:
                illegal_equal += 1
            tokens = text.split()
            all_tokens += len(tokens)
            alpha_tokens += sum(1 for token in tokens if token.isalpha() and len(token) >= 4)
    fraction = round(alpha_tokens / all_tokens, 4) if all_tokens else None
    findings.append(
        Finding(
            "NB-21",
            "Released illegal-sentence strings are short fragments; the paper's whole-word example was not found",
            _status_from_bool(illegal_total > 0 and illegal_equal == 0),
            [
                f"HotpotQA illegal fragments inspected: {illegal_total}.",
                f"Fragments equal to the paper example {paper_example!r}: {illegal_equal}.",
                f"Share of whitespace tokens that are alphabetic and at least 4 characters: {fraction}.",
                "First fragment: "
                + repr(loaded["hotpotqa"][0]["illegal sentence noise"][0]["text"][:180]),
            ],
            "The paper describes sampling words from a model vocabulary. The repository has no such generator, and the stored strings are not that paper example. We use the stored strings and do not call them a verified rerun of the paper's generator.",
        )
    )

    datatype_hits = 0
    datatype_total = 0
    for row in loaded.get("hotpotqa", []):
        value = row.get("datatype noise")
        texts = value if isinstance(value, list) else [value]
        for text in texts:
            if not isinstance(text, str):
                continue
            datatype_total += 1
            lowered = text.lower()
            if "http" in lowered or "def " in lowered or "public class" in lowered or "print(" in lowered:
                datatype_hits += 1
    findings.append(
        Finding(
            "NB-22",
            "Datatype noise usually mixes prose with a URL or code, but the cue is not present in every HotpotQA string",
            "CONFIRMED" if datatype_total > 0 else "UNVERIFIED",
            [
                f"HotpotQA datatype strings with a URL or code cue (http, def, public class, or print): {datatype_hits} of {datatype_total}.",
            ],
            "Where the cue is present, the passage still contains the original answer, so stripping URLs or code would make it look like Relevant. The 91 strings without that cue need manual inspection before anyone claims every datatype example matches the paper's text-plus-URL-or-code rule.",
        )
    )
    return findings


def render_markdown(source_findings: list[Finding], data_findings: list[Finding], command: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    confirmed = [item for item in source_findings + data_findings if item.status == "CONFIRMED"]
    lines = [
        "# NoiserBench issues verified for NoiseDetectorRAG",
        "",
        "This note records what the official NoiserBench repository does and does not contain.",
        "It is generated by `scripts/audit_noiserbench.py`. Re-run that script rather than editing the counts by hand.",
        "",
        f"- Generated: {now}",
        f"- NoiserBench commit inspected: `{NOISERBENCH_COMMIT}`",
        f"- Command: `{command}`",
        f"- Confirmed issues: {len(confirmed)}",
        "",
        "## How to reproduce",
        "",
        "```bash",
        "git clone --depth 1 https://github.com/jinyangwu/NoiserBench.git /tmp/NoiserBench",
        "python scripts/download_noiserbench.py",
        "python scripts/audit_noiserbench.py --repo /tmp/NoiserBench --data data/raw/noiserbench",
        "```",
        "",
        "The script is read-only. It does not change the NoiserBench checkout.",
        "",
        "## Confirmed issues",
        "",
        "| ID | Issue | What we checked |",
        "| --- | --- | --- |",
    ]
    for item in confirmed:
        evidence = " ".join(item.evidence)
        evidence = " ".join(evidence.split())
        evidence = evidence.replace("|", "/")
        if len(evidence) > 240:
            evidence = evidence[:237] + "..."
        lines.append(f"| {item.issue_id} | {item.title} | {evidence} |")
    lines.extend(["", "## Why this matters for our dataset", ""])
    lines.append(
        "NoiserBench is a RAG evaluation release. A record holds a question, a golden passage, "
        "and several noise fields side by side. It is not an eight-class training set with a "
        "label column or an official train/validation/test split. Our classification rows are "
        "derived from those fields, and each derived row keeps the source file, record id, field "
        "name, and commit."
    )
    lines.extend(["", "## Full evidence", ""])
    for item in source_findings + data_findings:
        lines.append(f"### {item.issue_id}. {item.title}")
        lines.append("")
        lines.append(f"Status: {item.status}")
        lines.append("")
        for evidence in item.evidence:
            lines.append(f"- {evidence}")
        lines.append("")
        lines.append(item.impact)
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/tmp/NoiserBench-inspect"))
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "noiserbench")
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs" / "noiserbench_issues.md")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs" / "dataset_reports" / "noiserbench_audit.json",
    )
    args = parser.parse_args()
    source_findings = check_source(args.repo)
    data_findings = check_data(args.data)
    command = " ".join(sys.argv)
    payload = {
        "noiserbench_commit": NOISERBENCH_COMMIT,
        "repo": str(args.repo),
        "data": str(args.data),
        "findings": [asdict(item) for item in source_findings + data_findings],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(
        render_markdown(source_findings, data_findings, command),
        encoding="utf-8",
    )
    confirmed = sum(item.status == "CONFIRMED" for item in source_findings + data_findings)
    unverified = sum(item.status == "UNVERIFIED" for item in source_findings + data_findings)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    print(f"Confirmed {confirmed}; unverified {unverified}")
    return 0 if unverified == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
