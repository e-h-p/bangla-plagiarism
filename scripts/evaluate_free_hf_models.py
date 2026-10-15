from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation_common import (
    ScoreWorkbookConfig,
    build_combined_scores,
    evaluate_scores,
    load_base_testcases,
    normalize_model_scores,
    summary_row,
    write_evaluation_outputs,
)


SCRIPT_DIR = Path(__file__).resolve().parent.parent
BASE_TESTCASES = SCRIPT_DIR / "dataset" / "banglaplagpair1k_base.xlsx"
OUTPUT_ROOT = SCRIPT_DIR / "free llm evualation"
COMBINED_OUTPUT_DIR = OUTPUT_ROOT / "combined_free_hf_model_evaluation"


MODELS = [
    ScoreWorkbookConfig(
        model_name="Qwen2.5 0.5B Instruct",
        slug="qwen2_5_0_5b_instruct",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Qwen2_5_0_5B_Score",),
        metric_column="Qwen2_5_0_5B_Instruct_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="Qwen2.5 1.5B Instruct",
        slug="qwen2_5_1_5b_instruct",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Qwen2.5-1.5B-Instruct",),
        metric_column="Qwen2_5_1_5B_Instruct_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="Gemma 2 2B IT",
        slug="gemma_2_2b_it",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Gemma_2_2B_Score",),
        metric_column="Gemma_2_2B_IT_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="Gemma 2 9B IT",
        slug="gemma_2_9b_it",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Gemma_2_9B_Score",),
        metric_column="Gemma_2_9B_IT_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="Aya Expanse 8B",
        slug="aya_expanse_8b",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Aya_Expanse_8B_Score",),
        metric_column="Aya_Expanse_8B_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="Llama 3.2 3B Instruct",
        slug="llama_3_2_3b_instruct",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Llama_3_2_3B_Score",),
        metric_column="Llama_3_2_3B_Instruct_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="Llama 3 8B Bangla 4bit",
        slug="llama_3_8b_bangla_4bit",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("Llama_3_8B_Bangla_4bit_Score",),
        metric_column="Llama_3_8B_Bangla_4bit_Score",
        detection_column=None,
        id_column="pair_id",
    ),
    ScoreWorkbookConfig(
        model_name="BanglaLLama 3 8B Instruct",
        slug="banglallama_3_8b_instruct",
        input_file=OUTPUT_ROOT / "combined_1000_testcases2.xlsx",
        score_columns=("BanglaLLama_3_8B_Score",),
        metric_column="BanglaLLama_3_8B_Score",
        detection_column=None,
        id_column="pair_id",
    ),
]


def write_model_outputs(
    config: ScoreWorkbookConfig,
    labeled_scores: pd.DataFrame,
    class_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    write_evaluation_outputs(
        output_dir=OUTPUT_ROOT / f"{config.slug}_evaluation",
        labeled_scores=labeled_scores,
        class_df=class_df,
        sweep_df=sweep_df,
        best_df=best_df,
        metrics=[config.metric_column],
        interactive_filename=f"interactive_threshold_{config.slug}.xlsx",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate free Hugging Face LLM similarity scores individually and as one combined report."
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only generate the combined free-HF evaluation folder without rewriting individual model workbooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base_testcases(BASE_TESTCASES)
    model_scores: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for config in MODELS:
        scores, filled_missing, used_score_columns = normalize_model_scores(base, config)
        model_scores[config.slug] = scores

        if args.combined_only:
            continue

        labeled_scores, class_df, sweep_df, best_df = evaluate_scores(scores, [config.metric_column])
        write_model_outputs(config, labeled_scores, class_df, sweep_df, best_df)

        output_dir = OUTPUT_ROOT / f"{config.slug}_evaluation"
        summary_rows.append(summary_row(config, class_df, best_df, output_dir.relative_to(SCRIPT_DIR)))
        print(f"Wrote {config.model_name} evaluation to {output_dir}")
        if filled_missing:
            print(f"  Filled {filled_missing} missing score(s) using the median from {used_score_columns}")

    combined_scores = build_combined_scores(base, MODELS, model_scores)
    combined_metrics = [config.metric_column for config in MODELS]
    labeled_scores, class_df, sweep_df, best_df = evaluate_scores(combined_scores, combined_metrics)
    write_evaluation_outputs(
        output_dir=COMBINED_OUTPUT_DIR,
        labeled_scores=labeled_scores,
        class_df=class_df,
        sweep_df=sweep_df,
        best_df=best_df,
        metrics=combined_metrics,
        interactive_filename="interactive_threshold_combined_free_hf_models.xlsx",
    )
    print(f"Wrote combined free-HF evaluation to {COMBINED_OUTPUT_DIR}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = OUTPUT_ROOT / "free_hf_model_evaluation_summary"
        summary.to_csv(summary_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(summary_path.with_suffix(".xlsx"), engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)

        print()
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
