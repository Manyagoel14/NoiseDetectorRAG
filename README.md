# NoiseDetectorRAG

Noise detection and pairwise noise-interaction analysis for retrieval-augmented generation.

The current milestone builds a documented eight-class question–passage dataset from the released NoiserBench files. It does not train a detector and it does not modify the NoiserBench repository.

## Show the NoiserBench issues

`docs/noiserbench_issues.md` is the note for the course report. Regenerate it with:

```bash
git clone --depth 1 https://github.com/jinyangwu/NoiserBench.git /tmp/NoiserBench
python scripts/download_noiserbench.py
python scripts/audit_noiserbench.py --repo /tmp/NoiserBench --data data/raw/noiserbench
```

The audit script is read-only. Twenty-two checks were confirmed against commit `447bee38d91996615ecc85c4da4a5b204f6c9837`.

## Rebuild the derived dataset

```bash
python scripts/download_noiserbench.py
python scripts/inspect_dataset.py
python scripts/run_dataset_pipeline.py
python scripts/validate_dataset.py
python -m unittest tests.test_dataset tests.test_audit_fixture
```

Raw JSON files are downloaded into `data/raw/noiserbench/` and are not committed. The derived JSONL is `data/processed/classification.jsonl`, also untracked. Checksums, reports, and plots are committed.

## What the derived set contains

HotpotQA supplies the seven passage classes. PriorQA supplies Prior Noise only. One passage is kept per parent question and class. Orthographic noise uses ratio 0.2. PriorQA counterfactual passages are held out. The split is 80/10/10 by normalized question, seed 42.

Measured result: 4,000 pairs, 500 per class, with 400/50/50 pairs per class in train/validation/test. Question overlap across splits was 0. See `PROGRESS.md` for the passage-title overlap that this split does not remove.
