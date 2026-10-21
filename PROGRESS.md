# Progress

## Objective and scope

Build an eight-class question–passage dataset for noise detection, with inspection, cleaning logs, class-distribution plots, and leakage checks. Detector training and RAG experiments are out of scope until this dataset is accepted.

## Approved roadmap

Milestone 1 from the audit plan: skeleton, pinned NoiserBench download, inspection, one pair per class, plots, parent-question splits, and a manual sample.

## Current status

COMPLETED for milestone 1. Training is NOT STARTED.

## Repository audit

NoiserBench commit `447bee38d91996615ecc85c4da4a5b204f6c9837` was inspected read-only. The write-up is `docs/noiserbench_issues.md`.

Command:

```bash
python scripts/audit_noiserbench.py --repo /tmp/NoiserBench-inspect --data data/raw/noiserbench
```

Result on 21 October 2026: 22 checks CONFIRMED, 0 UNVERIFIED. The script exited 0.

## Environment

This run used 4 CPU cores, 16 GB RAM, and no NVIDIA GPU. Python 3.12. Installed for this milestone: matplotlib 3.11.2 and PyYAML 6.0.1. Transformers was not installed, so sequence lengths are whitespace tokens.

## Dataset sources

Downloaded from the pinned commit into `data/raw/noiserbench/`. Bytes and SHA-256 are in `data/manifests/noiserbench_checksums.json`. The JSON files are gitignored.

| File | Records |
| --- | --- |
| nq | 807 |
| rgb | 300 |
| hotpotqa | 500 |
| 2wikimqa | 500 |
| bamboogle | 125 |
| strategyqa | 500 |
| tempqa | 186 |
| priorqa | 500 |

Total raw records: 3,418. None of these files has a `label` or `split` field. HotpotQA here is 500 rows, not the official 90,447-row distractor training set.

## Cleaning and exclusions

Derived sources: HotpotQA and PriorQA. Other files were inspected only, because every class already had 50 test pairs.

Rules and holdouts are in `docs/preprocessing_rules.md`.

Exclusion log: `outputs/dataset_reports/exclusion_log.csv`. It has a header and zero data rows. No HotpotQA or PriorQA candidate was dropped.

Held out by policy, not because the text was invalid:

- 500 second counterfactual passages on HotpotQA
- 500 counterfactual passages and 500 second counterfactual passages on PriorQA
- 4,500 orthographic strings at ratios other than 0.2

Stored inventory before that policy, from `outputs/dataset_reports/dataset_decision.json`:

| Class | Stored passages |
| --- | --- |
| Relevant | 500 |
| Semantic Noise | 5,000 |
| Datatype Noise | 500 |
| Illegal-Sentence Noise | 5,000 |
| Counterfactual Noise | 1,000 |
| Supportive Noise | 4,989 |
| Orthographic Noise | 5,000 |
| Prior Noise | 500 |

The largest stored class is 10 times the smallest. After the one-passage rule, every class has 500 rows, so the majority-to-minority ratio is 1. No oversampling was applied.

## Quality checks

From `outputs/dataset_reports/qc_report.json`:

- Relevant rows whose passage lacks the gold answer string: 0 of 500
- Supportive rows that still contain the gold answer string: 0 of 500
- Prior rows whose gold answer is not “The question is factually incorrect”: 0 of 500
- Parents whose two classes share identical passage text: 0
- Orthographic similarity to `gold_text` on the first 500 characters: min 0.014, median 0.551, max 0.9939
- Illegal-sentence fragments on the selected HotpotQA index: 30 whitespace tokens in every row

The orthographic maximum of 0.9939 means ratio 0.2 still leaves at least one passage nearly unchanged on that prefix. Those rows were kept and flagged rather than deleted.

## Splits and leakage

Seed 42. Ratios 0.8 / 0.1 / 0.1. Grouping key: normalized question. 1,000 question groups, 0 groups covering more than one parent.

| Split | Parents | Pairs per class |
| --- | --- | --- |
| train | 800 | 400 |
| validation | 100 | 50 |
| test | 100 | 50 |

`python scripts/validate_dataset.py` wrote `outputs/dataset_reports/leakage_report.json` with `ok: true`, 0 parent leaks, and 0 question leaks.

Passage overlap is reported and does not fail the check: 82 passage titles and 49 normalized passage texts appear in more than one split. A later split can group questions that share a title. That change is waiting for approval because it will shrink and unbalance the splits.

## Plots and reports

- `outputs/plots/class_counts_inventory.png`
- `outputs/plots/class_counts_raw.png`
- `outputs/plots/class_counts_cleaned.png`
- `outputs/plots/class_proportions_cleaned.png`
- `outputs/plots/class_counts_by_split.png`
- `outputs/plots/length_by_class_raw.png`
- `outputs/plots/length_by_class_cleaned.png`
- `outputs/dataset_reports/raw_inspection.md`
- `outputs/dataset_reports/manual_sample.jsonl` (HotpotQA ids 0–19, all seven passage classes, plus PriorQA ids 0–4)

## Detector and RAG

NOT STARTED.

## Tests

Command: `python -m unittest tests.test_dataset tests.test_audit_fixture`

Result: Ran 9 tests, OK. These tests use synthetic rows. They do not replace the full-data pipeline run above.

## Known limitations

- Labels are derived from fields. They are not an official eight-class annotation.
- Illegal-sentence and orthographic generators are not in the NoiserBench code, so those strings were reused, not regenerated.
- 91 of 500 HotpotQA datatype strings lack an `http`, `def`, `public class`, or `print` cue.
- Whitespace-token counts are not RoBERTa token counts. HotpotQA has 3 stored passages over 512 whitespace tokens and 97 over 256.
- 2WikiMultihopQA has 6 duplicate normalized questions. It is not in the derived split.
- Title and passage overlap across splits is real and unresolved.

## Next decision

Approve whether to keep this question-level split or regroup questions that share a Wikipedia title before any training.
