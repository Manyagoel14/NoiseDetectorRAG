# What each file is, and how to run the code yourself

Yes, the JSON, CSV, and chart files were produced by running the Python scripts. They were not typed in by hand.

The scripts read the downloaded NoiserBench files, counted the rows, built the labelled examples, and saved the results. A few notes, such as `PROGRESS.md` and `docs/dataset_schema.md`, were written afterwards to explain those results. The numbers in them come from the script outputs.

If you clone the GitHub repository, you will see the reports and charts. You will not see the big raw data files or the final labelled file. Those are left off GitHub on purpose because they are large. Run the steps below on your own computer and the scripts will create them again. With the same settings, the counts should match.

You need Python 3.10 or newer. From the project folder, install the two libraries the charts need:

```bash
python -m pip install matplotlib pyyaml
```

Run every command from the project folder, the folder that contains `scripts/`.

## Steps you can run

Do them in this order. Each step is a separate script. Read the files it creates before you start the next step.

### Step 1. Download the original files

```bash
python scripts/download_noiserbench.py
```

This downloads eight JSON files from the NoiserBench project at one fixed version. It does not change that project.

You get:

- `data/raw/noiserbench/` — the original files. One file per dataset. Each row is still a whole question with many noise fields packed together, not one label.
- `data/manifests/noiserbench_checksums.json` — the size of each file and a fingerprint, so you can check the download was not cut off.

### Step 2. Check the problems in NoiserBench

This step needs a copy of the NoiserBench code as well as the data from step 1.

```bash
git clone --depth 1 https://github.com/jinyangwu/NoiserBench.git /tmp/NoiserBench
python scripts/audit_noiserbench.py --repo /tmp/NoiserBench --data data/raw/noiserbench
```

You get:

- `docs/noiserbench_issues.md` — a plain report of the problems the script found. This is the file to show a professor.
- `outputs/dataset_reports/noiserbench_audit.json` — the same findings in a machine-readable form.

The script only reads NoiserBench. It does not edit it.

### Step 3. Count what is in the raw files

```bash
python scripts/inspect_dataset.py
```

This does not delete anything. It only counts.

You get:

- `outputs/dataset_reports/raw_inspection.json` — full counts: how many rows, empty questions, duplicate questions, how long the passages are.
- `outputs/dataset_reports/raw_inspection.md` — the same counts in shorter sentences.

### Step 4. Build our labelled dataset and the split

```bash
python scripts/build_classification_set.py
```

This reads the raw files and writes our own examples. One row becomes one question plus one passage plus one class. The raw files stay unchanged.

You get:

- `data/processed/classification.jsonl` — the dataset itself. One example per line. This file is not uploaded to GitHub.
- `outputs/dataset_reports/exclusion_log.csv` — rows we refused to keep, and why. In the run already done, this file has only a header, because nothing was refused.
- `outputs/dataset_reports/dataset_decision.json` — which source files were used, how many examples each class has, and the train, validation, and test sizes.
- `outputs/dataset_reports/qc_report.json` — quality checks, such as whether a “relevant” passage actually contains the answer.
- `outputs/dataset_reports/leakage_report.json` — whether the same question landed in two splits. `ok` means the question check passed.
- `outputs/dataset_reports/manual_sample.jsonl` — a small set of examples you can open and read. It has the first 20 HotpotQA questions and the first 5 PriorQA questions.
- `outputs/dataset_reports/candidate_lengths.json` — the length of each passage before the charts are drawn. The chart script needs this file.

### Step 5. Draw the charts

```bash
python scripts/analyze_class_distribution.py
```

This only reads the files from step 4. It does not make up the bar heights.

You get these pictures in `outputs/plots/`:

- `class_counts_inventory.png` — how many passages were stored in the original files. Some classes have about ten times as many as others.
- `class_counts_raw.png` — one possible passage per question, before the eligibility checks.
- `class_counts_cleaned.png` — the final dataset, one example per class per question.
- `class_proportions_cleaned.png` — the same final counts shown as shares of the whole dataset.
- `class_counts_by_split.png` — train, validation, and test side by side.
- `length_by_class_raw.png` and `length_by_class_cleaned.png` — how long the passages are in each class.

### Step 6. Check the split again

```bash
python scripts/validate_dataset.py
```

This reads `data/processed/classification.jsonl` and writes `outputs/dataset_reports/leakage_report.json` again. If the same question is in two splits, the script stops with an error.

### Optional. Run the small tests

```bash
python -m unittest tests.test_dataset tests.test_audit_fixture
```

These tests use tiny made-up examples. They check that the rules behave as written. They do not replace steps 1 to 6.

There is also `python scripts/run_dataset_pipeline.py`. That one command runs steps 3, 4, and 5 together. Use the separate scripts when you want to look at each result before continuing.

## What the other files are

| File | What it is |
| --- | --- |
| `README.md` | Short project description and the main commands. |
| `PROGRESS.md` | A status log: what is finished, the real counts, and what is still open. |
| `configs/dataset.yaml` | The choices for the build: random seed 42, split 80/10/10, orthographic noise ratio 0.2, and which datasets to use first. |
| `docs/dataset_schema.md` | The meaning of each field in a labelled example. |
| `docs/preprocessing_rules.md` | What we change and what we deliberately leave alone. Spelling errors and broken sentences are kept. |
| `docs/noiserbench_issues.md` | The professor note from step 2. |
| `pyproject.toml` | Project name and the two libraries to install. |
| `.gitignore` | Tells Git to skip the big raw files, the labelled dataset, logs, and model checkpoints. |
| `scripts/` | The commands in the steps above. |
| `src/rag_noise/` | The Python code those scripts call. |
| `tests/` | The small automatic checks. |
| `data/raw/noiserbench/` | Original downloaded JSON. Not on GitHub. |
| `data/processed/classification.jsonl` | Our labelled dataset. Not on GitHub. Created by step 4. |
| `data/manifests/noiserbench_checksums.json` | Download fingerprints from step 1. |

## What a labelled line contains

Open `outputs/dataset_reports/manual_sample.jsonl` in a text editor. Each line is one example. The important fields are:

- `question` — the question.
- `passage_text` — the passage, copied as it was. It is not cleaned up.
- `label` — one of the eight classes.
- `source_dataset` — `hotpotqa` or `priorqa`.
- `parent_id` — which original question this example came from. All versions of that question stay in the same split.
- `split` — `train`, `validation`, or `test`.
- `provenance` — which original field and which NoiserBench version the text came from.

## What this project has not done

No detector has been trained. No question-answering experiment has been run. The next choice, before training, is whether questions that share the same Wikipedia page should be forced into the same split. Right now only the question text is kept together. The leakage report says 82 titles and 49 passages still appear in more than one split.
