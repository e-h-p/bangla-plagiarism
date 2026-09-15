# BanglaPlagPair-1K experiment suite

This directory contains the self-contained code used to produce and validate
the manuscript's evaluation artefacts. It has no runtime dependency on earlier
experiment folders.

## Included methodology components

| Component | Local file |
| --- | --- |
| Bangla preprocessing and six traditional metrics | `bangla_plagiarism_pipeline.py` |
| Lexical-normalization rules | `lexical_normalization.tsv` |
| Bangla stopword list | `bangla_stopwords.txt` |
| Threshold sweeps and classification metrics | `evaluation_common.py` |
| Shared paths, validation, and configuration | `pipeline_common.py` |

The local methodology files preserve the implementations used for the reported
experiments. Keeping them beside the stage scripts makes a future public
release reproducible without private sibling directories.

## Dataset safety

`BanglaPlagPair-1K.xlsx` is opened read-only. Its SHA-256
is recorded at step 1 and checked again during consolidation. Derived files are
written under the relevant result directories; the polished dataset preserves
the original passages in `text1_raw` and `text2_raw`.

## Pipeline stages

| Step | Script | Main output |
| --- | --- | --- |
| 1 | `step1_build_base_dataset.py` | `dataset/banglaplagpair1k_base.*` |
| 2 | `step2_score_raw_dataset.py` | raw traditional scores and evaluation |
| 3 | `step3_polish_and_score.py` | polished text, scores, and evaluation |
| 4 | `step4_run_banglabert_evaluation.py` | raw and polished BanglaBERT results |
| 5 | `step5_preprocessing_component_ablation.py` | component-wise ablation |
| 6 | `step6_consolidate_results.py` | consolidated results and validation |
| 7 | `step7_sentence_embedding_evaluation.py` | sentence-encoder evaluation |
| 8 | `step8_free_llm_evaluation.py` | open-weight LLM evaluation |
| 9 | `step9_prompt_engineering_evaluation.py` | prompt-strategy evaluation |
| 10 | `step10_paid_llm_evaluation.py` | paid-LLM evaluation |

Stages 7--10 consume archived score workbooks and can be run independently.
They validate pair identifiers, labels, text alignment, numeric score ranges,
and expected row counts before calculating results.

Run the complete sequence from this directory:

```bash
python run_all_experiments.py
```

## Fixed settings

- Traditional metrics: TF-IDF cosine, Jaccard, LCS, SequenceMatcher,
  edit-distance similarity, and word overlap.
- Full preprocessing: NFC normalization, lowercasing, digit normalization,
  punctuation removal, lexical mapping at confidence 0.85 or higher, stopword
  removal, and conservative stemming.
- BanglaBERT checkpoints: `csebuetnlp/banglabert` and
  `sagorsarker/bangla-bert-base`.
- Encoder pooling: attention-masked mean pooling followed by L2 normalization.
- Fixed threshold: 0.50.
- Threshold grid: 0.10 through 0.90 in increments of 0.05.
- Threshold tie-break: F1, accuracy, recall, then precision.
- Random seed: 42.
- Archived LLM scores are never imputed, clipped, renamed, or overwritten.

## Validation

`verify_dataset_consistency.py`, `verify_paid_llm_evaluation.py`, and
`verify_report_paid_llm_values.py` provide independent consistency checks.
The `validation/` outputs record row order, labels, source links, text
preservation, hashes, and agreement between consolidated and source results.
