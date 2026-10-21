# Preprocessing rules

Raw NoiserBench JSON is not edited. Derived rows live under `data/processed/`.

## Applied to every class

- Empty questions and empty passages are excluded and written to `outputs/dataset_reports/exclusion_log.csv`. The current HotpotQA and PriorQA build had no such rows.
- Exact duplicate `(parent, label, passage)` rows would be a validator error. None were emitted.
- `passage_text` is the released string. It is not lowercased, spell-corrected, or stripped of URLs, code, or unusual spacing.
- Unicode NFC is stored in `passage_text_nfc` only when it does not change orthographic or illegal-sentence text.
- Answer normalization is used only to test whether a gold answer occurs in a passage.

## Class-specific choices

- Relevant uses `gold_text` from the passage-level files.
- Semantic, illegal-sentence, and datatype rows use the first non-empty entry.
- Supportive rows skip passages that contain a gold answer under the normalized check.
- Counterfactual rows use `counterfactual noise`. The second counterfactual passage is counted as held out.
- Orthographic rows use ratio `0.2` and are dropped only if that string is identical to `gold_text`.
- Prior rows come only from PriorQA. Counterfactual passages in that file are held out so a false-premise question is not also labelled Counterfactual.

## Not done in this milestone

- No spell correction, grammar repair, or fact correction.
- No class balancing by copying validation or test rows. The one-passage rule already yields 500 rows per class.
- No tokenizer truncation. Whitespace-token thresholds are reported in `outputs/dataset_reports/raw_inspection.md`. They are not RoBERTa token counts.
- Other NoiserBench files were inspected and then left out of the derived set because HotpotQA and PriorQA already give at least 30 test pairs per class.
