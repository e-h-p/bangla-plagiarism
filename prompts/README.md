# Prompt catalog

## P0: shared semantic-similarity prompt

The exact P0 text is in `shared_similarity_prompt.txt`. It was used for the eight open/open-weight models and the eight paid routes. The same template is embedded in:

- `../generation_codes/open_llm_scoring_colab.py`
- `../generation_codes/paid_llm_scoring_colab.py`

## P1-P8: prompt-engineering variants

The exact, executable definitions are the `PROMPT_VARIANTS` dictionary in `../generation_codes/prompt_engineering_scoring_colab.py`:

| Code | Dictionary key | Short description |
| --- | --- | --- |
| P1 | `P1_FewShot_Anchored` | Four scoring anchors |
| P2 | `P2_Contradiction_Aware` | Checks contradictory factual direction |
| P3 | `P3_Hard_Negative_Aware` | Prioritizes false-positive avoidance |
| P4 | `P4_Bangla_Instruction` | Bangla-language instructions |
| P5 | `P5_NLI_Semantic` | Natural-language-inference framing |
| P6 | `P6_MultiDimensional` | Six internal comparison dimensions |
| P7 | `P7_Calibrated_Conservative` | Conservative score calibration |
| P8 | `P8_Direct_Short` | Minimal direct instruction |

The placeholders `{text1}` and `{text2}` are replaced at inference time. Every prompt requests a single score in `[0,1]`; the generator accepts Bangla or English digits and rejects missing or out-of-range values.
