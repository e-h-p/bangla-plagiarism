from __future__ import annotations

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


ROOT = Path(__file__).resolve().parent.parent
BASE_FILE = ROOT / "dataset" / "banglaplagpair1k_base.xlsx"
INPUT_FILE = (
    ROOT
    / "Sentence embedding models evualtion"
    / "combined_1000_testcases_ scores3.xlsx"
)
OUTPUT_ROOT = ROOT / "Sentence embedding models evualtion"


CONFIGS = [
    ScoreWorkbookConfig(
        model_name="LaBSE",
        slug="labse",
        input_file=INPUT_FILE,
        score_columns=("LaBSE_Score",),
        metric_column="LaBSE_Score",
        detection_column=None,
        id_column="pair_id",
        clip_scores=True,
    ),
    ScoreWorkbookConfig(
        model_name="Multilingual E5 Base",
        slug="e5_base",
        input_file=INPUT_FILE,
        score_columns=("E5_Base_Score",),
        metric_column="E5_Base_Score",
        detection_column=None,
        id_column="pair_id",
        clip_scores=True,
    ),
    ScoreWorkbookConfig(
        model_name="BGE-M3",
        slug="bge_m3",
        input_file=INPUT_FILE,
        score_columns=("BGE_M3_Score",),
        metric_column="BGE_M3_Score",
        detection_column=None,
        id_column="pair_id",
        clip_scores=True,
    ),
    ScoreWorkbookConfig(
        model_name="Multilingual MiniLM",
        slug="minilm_multilingual",
        input_file=INPUT_FILE,
        score_columns=("MiniLM_Multilingual_Score",),
        metric_column="MiniLM_Multilingual_Score",
        detection_column=None,
        id_column="pair_id",
        clip_scores=True,
    ),
]


def score_audit(input_file: Path, configs: list[ScoreWorkbookConfig]) -> pd.DataFrame:
    df = pd.read_excel(input_file, engine="openpyxl")
    rows = []
    for config in configs:
        raw = pd.to_numeric(df[config.metric_column], errors="coerce")
        rows.append(
            {
                "Model": config.model_name,
                "Metric": config.metric_column,
                "Rows": len(raw),
                "Missing": int(raw.isna().sum()),
                "Below_0": int((raw < 0).sum()),
                "Above_1": int((raw > 1).sum()),
                "Raw_Min": raw.min(),
                "Raw_Max": raw.max(),
                "Raw_Mean": raw.mean(),
                "Clipped_Min": raw.clip(0.0, 1.0).min(),
                "Clipped_Max": raw.clip(0.0, 1.0).max(),
                "Clipped_Mean": raw.clip(0.0, 1.0).mean(),
            }
        )
    return pd.DataFrame(rows)


def score_distribution_summary(scores: pd.DataFrame, configs: list[ScoreWorkbookConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        metric = config.metric_column
        for label_name, label_value in [("Positive", 1), ("Mismatch", 0)]:
            values = scores.loc[scores["Label"].eq(label_value), metric]
            rows.append(
                {
                    "Model": config.model_name,
                    "Metric": metric,
                    "Class": label_name,
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Std": values.std(ddof=0),
                    "Min": values.min(),
                    "Max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def category_error_analysis(
    scores: pd.DataFrame,
    configs: list[ScoreWorkbookConfig],
    fixed_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    thresholds = []
    for config in configs:
        fixed_threshold = float(fixed_df.loc[fixed_df["Metric"].eq(config.metric_column), "Threshold"].iloc[0])
        best_threshold = float(best_df.loc[best_df["Metric"].eq(config.metric_column), "Threshold"].iloc[0])
        thresholds.extend(
            [
                (config, "Fixed", fixed_threshold),
                (config, "Best-F1", best_threshold),
            ]
        )

    for config, setting, threshold in thresholds:
        metric = config.metric_column
        predicted = (scores[metric] >= threshold).astype(int)
        actual = scores["Label"].astype(int)

        for variant_type, group in scores.groupby("variant_type", sort=False):
            idx = group.index
            group_pred = predicted.loc[idx]
            group_actual = actual.loc[idx]
            tp = int(((group_pred == 1) & (group_actual == 1)).sum())
            fp = int(((group_pred == 1) & (group_actual == 0)).sum())
            tn = int(((group_pred == 0) & (group_actual == 0)).sum())
            fn = int(((group_pred == 0) & (group_actual == 1)).sum())
            rows.append(
                {
                    "Model": config.model_name,
                    "Metric": metric,
                    "Setting": setting,
                    "Threshold": threshold,
                    "Variant_Type": variant_type,
                    "Pairs": len(group),
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "Recall": tp / (tp + fn) if (tp + fn) else None,
                    "Specificity": tn / (tn + fp) if (tn + fp) else None,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    base = load_base_testcases(BASE_FILE)
    model_scores: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for config in CONFIGS:
        normalized, filled_missing, used_score_columns = normalize_model_scores(base, config)
        labeled, fixed, sweep, best = evaluate_scores(normalized, [config.metric_column])
        output_dir = OUTPUT_ROOT / f"{config.slug}_evaluation"
        write_evaluation_outputs(
            output_dir=output_dir,
            labeled_scores=labeled,
            class_df=fixed,
            sweep_df=sweep,
            best_df=best,
            metrics=[config.metric_column],
            interactive_filename=f"interactive_threshold_{config.slug}.xlsx",
        )
        row = summary_row(config, fixed, best, output_dir)
        row["Filled_Missing_Scores"] = filled_missing
        row["Used_Score_Columns"] = ", ".join(used_score_columns)
        summary_rows.append(row)
        model_scores[config.slug] = labeled

    combined_scores = build_combined_scores(base, CONFIGS, model_scores)
    combined_labeled, combined_fixed, combined_sweep, combined_best = evaluate_scores(
        combined_scores,
        [config.metric_column for config in CONFIGS],
    )
    combined_dir = OUTPUT_ROOT / "combined_sentence_embedding_evaluation"
    write_evaluation_outputs(
        output_dir=combined_dir,
        labeled_scores=combined_labeled,
        class_df=combined_fixed,
        sweep_df=combined_sweep,
        best_df=combined_best,
        metrics=[config.metric_column for config in CONFIGS],
        interactive_filename="interactive_threshold_sentence_embeddings.xlsx",
    )

    audit = score_audit(INPUT_FILE, CONFIGS)
    distribution = score_distribution_summary(combined_labeled, CONFIGS)
    category_errors = category_error_analysis(combined_labeled, CONFIGS, combined_fixed, combined_best)
    summary = pd.DataFrame(summary_rows)

    summary.to_csv(OUTPUT_ROOT / "sentence_embedding_model_evaluation_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUTPUT_ROOT / "sentence_embedding_score_audit.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(OUTPUT_ROOT / "sentence_embedding_score_distribution_summary.csv", index=False, encoding="utf-8-sig")
    category_errors.to_csv(OUTPUT_ROOT / "sentence_embedding_category_error_analysis.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_ROOT / "sentence_embedding_model_evaluation_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="model_summary", index=False)
        combined_fixed.to_excel(writer, sheet_name="threshold_0_50", index=False)
        combined_best.to_excel(writer, sheet_name="best_thresholds", index=False)
        audit.to_excel(writer, sheet_name="score_audit", index=False)
        distribution.to_excel(writer, sheet_name="score_distribution", index=False)
        category_errors.to_excel(writer, sheet_name="category_errors", index=False)

    print("Sentence embedding evaluation complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
