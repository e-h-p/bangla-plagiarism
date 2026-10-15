# Bangla plagiarism screening experiments

This package supports the MLWA manuscript **Specificity-Aware Intelligent Screening for Bangla Plagiarism: Evaluating Lexical, Embedding, and Large Language Models**. It provides preprocessing resources, deterministic evaluation code, generation templates, archived pair-level model scores, reference metrics and analysis figure code.

The task distinguishes meaning-preserving source-derived transformations from semantic mismatches on a controlled benchmark. Scores are screening evidence, not judgments of plagiarism misconduct.

## Dataset

**BanglaPlagPair-1K: A controlled Bangla text-pair dataset for plagiarism-oriented transformations and semantic non-equivalence**, Version 1, is available from [Mendeley Data](https://doi.org/10.17632/ws5c8ppx5m.1), under CC BY-SA 4.0 and the source-specific attribution/licensing conditions documented there.

Download `BanglaPlagPair-1K.xlsx` separately. The canonical dataset is distributed by Mendeley; score workbooks here also retain passage texts for output alignment. Required fields are `pair_id`, `text1`, `pair_type`, `binary_label`, `text2`, `source_link_text1` and `source_link_text2`.

The 1,000 pairs contain 500 positives (100 each of Morphological, Order, Synonymy, Lexical and Mixed) and 500 mismatches. The first 600 rows form 100 six-pair reference blocks. Negative groups have 100 shared-reference, 200 same-source and 200 cross-source pairs. Related references/source passages introduce dependence; keep groups disjoint when developing future train/test partitions.

## Methods

- **Preprocessing:** NFC/invisible-character cleanup, lowercasing, Bangla/Arabic-Indic digit normalization, punctuation/symbol removal, longest-match lexical mapping, stopword removal and conservative stemming. The lexical file has 162 entries; confidence filtering at 0.85 and normalized identity/duplicate removal leave 120 active mappings. The 73 stopword entries normalize to 71 unique words. Nine ablation settings cover raw text, each of seven components and the full pipeline.
- **Traditional metrics:** pair-fitted TF-IDF cosine, token-set Jaccard, character LCS divided by maximum string length, Python `SequenceMatcher.ratio()` with its default settings, normalized Levenshtein similarity and word-overlap coefficient. Token metrics retain their common tokenizer in the raw condition. Traditional scores are rounded to six decimals before evaluation.
- **Sentence encoders:** LaBSE, multilingual E5-base, BGE-M3 and multilingual MiniLM. Checkpoints are `sentence-transformers/LaBSE`, `intfloat/multilingual-e5-base`, `BAAI/bge-m3`, and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. E5 receives `query:` on both passages. Evaluation clips cosine values to [0,1]; exactly one archived LaBSE value is negative, below every evaluated threshold.
- **BanglaBERT baselines:** `csebuetnlp/banglabert` is an ELECTRA discriminator pretrained with replaced-token detection; `sagorsarker/bangla-bert-base` uses masked-language modeling. Both use attention-masked mean pooling and L2 normalization, maximum length 512 and batch size 8. The archived experiment uses raw text and the study's custom polishing, without the separate BUET model-card normalization pipeline. Raw and polished cosine scores are included for inference-free evaluation.
- **P0 open/open-weight LLMs:** Qwen2.5 0.5B/1.5B Instruct, Gemma 2 2B/9B IT, Aya Expanse 8B, Llama 3.2 3B Instruct, Llama 3 8B Bangla 4bit and BanglaLLama 3 8B Instruct. Generation uses 4-bit NF4 double quantization, FP16 computation, greedy decoding, a plain instruction/response wrapper and at most 15 new tokens.
- **Paid LLMs:** DeepSeek V4 Flash, GPT 5.6 Luna, Gemini 3.5 Flash Lite, Qwen 3.7 Flash, Claude Sonnet 5, Gemini 3.6 Flash, DeepSeek V4 Pro 0813 and Grok 4.3. Exact routes are in `generation_codes/paid_llm_scoring_colab.py`. The protocol requests temperature 0, Gemini 3.x minimal reasoning with returned reasoning excluded and 128 output tokens, and reasoning disabled with 12 output tokens on other routes.
- **Prompt engineering:** P1 few-shot anchored, P2 contradiction-aware, P3 hard-negative-aware, P4 Bangla-language rubric, P5 NLI semantic, P6 multidimensional, P7 conservative and P8 direct-short. Three selected P0 models yield 24 configurations. The generator uses the documented 4-bit setup, literal instruction/response wrapper, 4,096-token input limit and at most 12 generated tokens. Exact templates are indexed in `prompts/README.md` and executable in `generation_codes/prompt_engineering_scoring_colab.py`.

Descriptive overlap statistics use **whitespace-token sets**, separately from the normalized Jaccard evaluation metric. Shared-reference/same-source/cross-source means round to 0.38/0.06/0.01, all mismatches to 0.11, and Order/Morphological/Synonymy/Lexical/Mixed to 0.82/0.76/0.49/0.34/0.33.

## Evaluation protocol

`binary_label=1` is positive. Predict positive when `score >= threshold`. The fixed threshold is 0.50; the ascending grid is 0.10, 0.15, ..., 0.90 (17 points). Select by F1, accuracy, recall and precision, retaining the lowest threshold if all four tie. Metric tables store six decimals. Selection uses the same benchmark and is an in-sample diagnostic, not an independent test estimate.

Outputs include TP/FP/TN/FN, accuracy, precision, recall, F1, specificity, false-positive rate and balanced accuracy. Accuracy equals balanced accuracy on this balanced dataset. Analyses include preprocessing deltas, class-conditional distributions, category errors and threshold sensitivity. Missing, nonnumeric or infinite scores are rejected. Out-of-range LLM scores are rejected without imputation/clipping; sentence cosines use the documented clipping.

## Installation

Python 3.11 is recommended. The archived-score workflow needs no PyTorch, Transformers, GPU, API account or credentials.

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# macOS/Linux instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

`Levenshtein` accelerates exact edit distance; the code retains a Python fallback. Optional fresh model inference needs additional packages:

```bash
python -m pip install -r requirements-models.txt
```

Quantized generation requires a compatible accelerator/bitsandbytes setup. The original generation scripts read credentials from `HF_TOKEN` or `OPENROUTER_API_KEY`; no credential is included. Configure their workbook paths to disposable input copies: generation writes scores into the configured workbook.

## Reproduce analyses

From the package root, supply the downloaded unmodified Version 1 dataset:

```bash
python reproduce.py --dataset "path/to/BanglaPlagPair-1K.xlsx" --output "analysis_run" --figures
```

The output directory must not exist. The command stages separate copies of the code, dataset and archives there. It recomputes traditional scores and all nine preprocessing profiles, evaluates archived encoder/LLM scores, consolidates results and optionally renders figures. It compares **180** generated fixed/grid/best metric CSVs with verified reference tables, checking confusion counts exactly and other values to archived rounding precision. `analysis_run/reproduction_validation.json` records comparisons and original-input preservation. No LLM API is called.

Outputs under `analysis_run/` include:

- `results/`: consolidated traditional/BanglaBERT results, deltas and distributions.
- `preprocessing_component_ablation/`: nine-profile results, summaries and plots.
- `Sentence embedding models evualtion/`: four encoder evaluations, score audit, distributions and category errors.
- `free llm evualation/` and `paid llm evualation/`: eight P0 evaluations per group and summaries.
- `prompt_engineering_evaluation/`: 24 configurations and strategy averages.
- `report/figures/`: analysis plots when `--figures` is supplied.
- `validation/`: dataset alignment and integrity reports.

Historical directory spellings are retained. Numbered scripts can run individually, but write under their parent project directory. Use the root command's new output tree to preserve existing analyses.

Optional fresh BanglaBERT encoding instead of archived cosines:

```bash
python reproduce.py --dataset "path/to/BanglaPlagPair-1K.xlsx" --output "encoder_rerun" --regenerate-banglabert
```

This downloads checkpoints, requires the model dependencies and still compares metrics with the references. Changed model/library revisions may cause comparison failures. Sentence-encoder and LLM generation are separate, opt-in operations through `generation_codes/`, never invoked by the archived-score command.

## Package structure

| Path | Contents |
| --- | --- |
| `reproduce.py` | Isolated runner and reference-result comparison |
| `scripts/` | Stages 1-10, restored implementations, preprocessing resources, workbook builder, archived-BanglaBERT evaluator and plot generation |
| `generation_codes/` | Original optional model-generation scripts |
| `prompts/` | P0 text and P1-P8 catalog |
| `sentence_embedding_scores/` | Original four-encoder workbook |
| `open_llm_scores/`, `paid_llm_scores/` | Original eight-model/route workbooks |
| `prompt_engineering_scores/` | Original three-model P1-P8 workbooks |
| `banglabert_scores/normal/`, `banglabert_scores/polished/` | Unchanged pair-level cosine CSVs and archived diagnostics |
| `reference_results/` | Verified fixed, swept and selected-threshold metric CSVs |
| `requirements.txt`, `requirements-models.txt` | Analysis and optional inference dependencies |
| `SHA256SUMS.txt` | Prepared public-file checksums, excluding itself |

The paid-score workbook had embedded absolute folder metadata removed from its Excel container. Every worksheet cell, score and other archive member remains unchanged.

`requirements-validated.txt` records the resolved dependencies from the clean Python 3.11 validation environment; use it instead of the version ranges when an identical analysis environment is wanted.

## Known reproducibility limits

Recomputing metrics from archived scores differs from regenerating the scores. Complete raw LLM responses, immutable checkpoint revisions, per-request hosted-route metadata and a locked original inference environment were not retained for every run. Greedy decoding or temperature 0 cannot ensure identical outputs across changed runtimes/providers. Generation scripts document the available protocol but cannot restore missing historical records.

Thresholds, model comparisons and prompt selection all use the same controlled benchmark. Near-perfect/zero-error configurations describe this dataset, without establishing unseen-domain performance. The analysis reports point estimates, without confidence intervals or paired significance tests. The separate Mendeley release maintains the canonical dataset and anonymous external judgments. Third-party passage licensing remains applicable. Dependency ranges describe a working analysis environment; resolved versions may differ from the original inference environment.
