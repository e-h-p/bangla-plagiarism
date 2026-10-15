# Analysis entry points

Use the root `reproduce.py` command documented in `../README.md`. It stages a copy of these scripts, the separately supplied dataset and unchanged archived scores in a new output directory before execution.

Stages 1-3 regenerate dataset aliases, raw scores and polished scores. Stage 5 regenerates all nine preprocessing profiles. `evaluate_archived_banglabert.py` evaluates the archived raw/polished encoder scores. Stage 6 consolidates the traditional/BanglaBERT results. Stages 7-10 evaluate the archived sentence-embedding, P0 open-model, P1-P8 and paid-model scores. `generate_report_figures.py` renders the resulting analysis plots.

`step4_run_banglabert_evaluation.py` is the optional fresh-inference alternative, enabled by `reproduce.py --regenerate-banglabert`; it requires the model dependencies. The restored `evaluate_sentence_embedding_models.py`, `evaluate_free_hf_models.py`, and `run_preprocessing_component_ablation.py` are the implementations called by their numbered stage wrappers. `evaluation_common.py` and `build_interactive_threshold_workbook.py` provide shared metric calculations and interactive Excel dashboards.

Running a numbered script directly writes into its parent project directory. Run it only in the staged output tree if existing results must be preserved. The root entry point creates that tree and refuses to reuse an existing output directory.
