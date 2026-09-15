# Bangla Text Reuse Screening Experiments

This repository contains prompts, scoring code, evaluation and analysis scripts, and archived score files used for Bangla plagiarism/text-reuse screening experiments.

## Dataset

**Dataset name:**  
**BanglaPlagPair-1K: A controlled Bangla text-pair dataset for plagiarism-oriented transformations and semantic non-equivalence**

**Dataset DOI:**  
https://doi.org/10.17632/ws5c8ppx5m.1

The dataset is hosted separately on Mendeley Data. It is not duplicated in this repository.

## Repository Contents

| Path | Description |
| --- | --- |
| `generation_codes/` | Colab/Python scripts used for generating model scores. |
| `open_llm_scores/` | Archived pair-level score workbook for open and open-weight LLM experiments. |
| `paid_llm_scores/` | Archived pair-level score workbook for hosted/paid LLM experiments. |
| `prompts/` | Shared prompt definitions and prompt-engineering variants used in the experiments. |
| `prompt_engineering_scores/` | Archived score workbooks for prompt-engineering experiments. |
| `scripts/` | Evaluation and analysis scripts used to support the reported experiments. |
| `sentence_embedding_scores/` | Archived score workbook for sentence-embedding experiments. |
| `requirements.txt` | Basic Python dependencies used by the evaluation and analysis scripts. |
| `requirements-models.txt` | Additional dependencies required for model score-generation scripts. |

## Experimental Scope

The repository supports the experiments reported in the associated study, including:

- traditional lexical and string-based similarity methods;
- Bangla-aware preprocessing experiments;
- sentence-embedding model evaluation;
- unadapted BanglaBERT cosine-similarity baselines;
- open and open-weight LLM scoring;
- hosted/paid LLM scoring;
- prompt-engineering experiments;
- threshold-based evaluation;
- specificity and false-positive analysis.

The archived score files contain the pair-level model outputs used in the reported comparisons.

## Installation

Install the basic dependencies with:

```bash
pip install -r requirements.txt