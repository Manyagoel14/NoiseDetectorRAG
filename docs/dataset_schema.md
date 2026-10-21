# Derived classification schema

Each line of `data/processed/classification.jsonl` is one question–passage pair. The label is assigned by this project from a NoiserBench field. The raw files do not contain that label.

| Field | Meaning |
| --- | --- |
| `example_id` | Stable hash of parent, label, field, variant, and passage text |
| `parent_id` | `{source}:{source id}`, for example `hotpotqa:0` |
| `source_dataset` | `hotpotqa` or `priorqa` in the current build |
| `source_example_id` | Integer id inside the released JSON file |
| `question` | Question text, unchanged |
| `normalized_question` | Lowercased question with collapsed whitespace, used only for splitting |
| `passage_title` | Wikipedia title when the field has one, otherwise an empty string |
| `passage_text` | Passage text, unchanged |
| `passage_text_nfc` | NFC copy, or null when NFC would change orthographic or illegal-sentence text |
| `label` | One of the eight class names |
| `label_id` | 0–7 in the fixed order below |
| `gold_answers` | Answer strings. Booleans become `true` or `false` |
| `field_name` | Released field the passage came from |
| `variant_index` | Index inside a list-valued field. `0` for a string field |
| `orthographic_ratio` | `"0.2"` for orthographic rows, otherwise null |
| `split` | `train`, `validation`, or `test` |
| `provenance` | Commit, source file, and selection rule |
| `qc` | Answer-containment and orthographic-similarity flags |

Label order: Relevant, Semantic Noise, Datatype Noise, Illegal-Sentence Noise, Counterfactual Noise, Supportive Noise, Orthographic Noise, Prior Noise.

Selection rules are stored on each row. The important ones are: supportive noise keeps the first passage whose normalized text does not contain a gold answer; orthographic noise keeps ratio 0.2; Prior Noise is the PriorQA question paired with `gold_text`.
