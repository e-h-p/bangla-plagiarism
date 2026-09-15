# -*- coding: utf-8 -*-
"""Evaluate supplied paid-LLM scores with the archived study protocol.

The implementation and workbook builder are included locally. In particular,
this stage uses:

* a fixed decision threshold of 0.50;
* the inclusive 0.10--0.90 sweep in 0.05 increments;
* ``score >= threshold`` as the positive decision rule;
* F1 -> accuracy -> recall -> precision for threshold tie-breaking; and
* the same seven files per model plus the combined evaluation and summary.

The scored input workbook and canonical dataset are opened read-only.  Missing,
non-numeric or out-of-range scores are rejected instead of being imputed or
clipped, so the supplied score columns remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parent
from build_interactive_threshold_workbook import build_workbook


BASE_DATASET = WORK_ROOT / "dataset" / "banglaplagpair1k_base.xlsx"
SCORED_WORKBOOK = WORK_ROOT / "paid llm evualation" / "combined_1000_testcases.xlsx"
OUTPUT_ROOT = WORK_ROOT / "paid llm evualation"
COMBINED_OUTPUT_DIR = OUTPUT_ROOT / "combined_llm_model_evaluation"

FIXED_THRESHOLD = 0.50
THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90,
]
OUTPUT_BASE_COLUMNS = [
    "pair_no", "dataset", "source_case_no", "variant_type", "text1", "text2",
]


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    slug: str
    score_column: str

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / f"{self.slug}_evaluation"


MODELS = [
    ModelConfig("DeepSeek V4 Flash", "deepseek_v4_flash", "DeepSeek_V4_Flash_Score"),
    ModelConfig("GPT 5.6 Luna", "gpt_5_6_luna", "GPT_5_6_Luna_Score"),
    ModelConfig(
        "Gemini 3.5 Flash Lite",
        "gemini_3_5_flash_lite",
        "Gemini_3_5_Flash_Lite_Score",
    ),
    ModelConfig("Qwen 3.7 Flash", "qwen_3_7_flash", "Qwen_3_7_Flash_Score"),
    ModelConfig("Claude Sonnet 5", "claude_sonnet_5", "Claude_Sonnet_5_Score"),
    ModelConfig("Gemini 3.6 Flash", "gemini_3_6_flash", "Gemini_3_6_Flash_Score"),
    ModelConfig(
        "DeepSeek V4 Pro 0813",
        "deepseek_v4_pro_0813",
        "DeepSeek_V4_Pro_0813_Score",
    ),
    ModelConfig("Grok 4.3", "grok_4_3", "Grok_4_3_Score"),
]


def load_base_testcases() -> pd.DataFrame:
    base = pd.read_excel(BASE_DATASET, engine="openpyxl")
    required = ["pair_no", "pair_id", "variant_type", "text1", "text2", "Label"]
    missing = [column for column in required if column not in base.columns]
    if missing:
        raise ValueError(f"Canonical base workbook is missing columns: {missing}")
    if not base["pair_no"].is_unique or not base["pair_id"].is_unique:
        raise ValueError("Canonical pair_no and pair_id values must both be unique.")

    derived_label = base["variant_type"].astype(str).str.lower().ne("mismatch").astype(int)
    supplied_label = pd.to_numeric(base["Label"], errors="raise").astype(int)
    if not supplied_label.equals(derived_label):
        raise ValueError("Canonical Label disagrees with the historical mismatch labelling rule.")
    if "binary_label" in base.columns:
        binary_label = pd.to_numeric(base["binary_label"], errors="raise").astype(int)
        if not binary_label.equals(supplied_label):
            raise ValueError("Canonical Label and binary_label columns disagree.")

    prepared = base.copy()
    prepared["dataset"] = derived_label.map({1: "textcase", 0: "mismatch"})
    prepared["source_case_no"] = prepared["pair_no"]
    prepared = prepared[OUTPUT_BASE_COLUMNS + ["pair_id", "Label"]].copy()
    return prepared


def load_and_validate_scores(base: pd.DataFrame) -> pd.DataFrame:
    scores = pd.read_excel(SCORED_WORKBOOK, sheet_name="Sheet1", engine="openpyxl")
    required = ["pair_id", "text1", "text2"] + [model.score_column for model in MODELS]
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise ValueError(f"Paid-LLM workbook is missing columns: {missing}")
    if not scores["pair_id"].is_unique:
        raise ValueError("Paid-LLM pair_id values must be unique.")

    keep = scores[required].copy()
    for config in MODELS:
        numeric = pd.to_numeric(keep[config.score_column], errors="coerce")
        if numeric.isna().any():
            count = int(numeric.isna().sum())
            raise ValueError(f"{config.score_column} contains {count} missing/non-numeric scores.")
        outside = ~numeric.between(0.0, 1.0, inclusive="both")
        if outside.any():
            ids = keep.loc[outside, "pair_id"].tolist()
            raise ValueError(f"{config.score_column} has out-of-range scores at pair_id {ids[:10]}.")
        keep[config.score_column] = numeric

    audit = base[["pair_id", "text1", "text2"]].merge(
        keep[["pair_id", "text1", "text2"]],
        on="pair_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_base", "_paid"),
        indicator=True,
    )
    if not audit["_merge"].eq("both").all():
        raise ValueError("Paid-LLM pair_id values do not match the canonical benchmark.")
    if not audit["text1_base"].astype(str).equals(audit["text1_paid"].astype(str)):
        raise ValueError("Paid-LLM text1 values do not match the canonical benchmark.")
    if not audit["text2_base"].astype(str).equals(audit["text2_paid"].astype(str)):
        raise ValueError("Paid-LLM text2 values do not match the canonical benchmark.")

    return keep.drop(columns=["text1", "text2"])


def classification_metrics(
    scores: pd.DataFrame,
    threshold: float,
    metrics: list[str],
) -> pd.DataFrame:
    """Historical paid-LLM metric table, retained byte-for-schema compatibility."""
    rows: list[dict[str, object]] = []
    actual = scores["Label"].astype(int)

    for metric in metrics:
        predicted = (scores[metric] >= threshold).astype(int)
        tp = int(((predicted == 1) & (actual == 1)).sum())
        fp = int(((predicted == 1) & (actual == 0)).sum())
        tn = int(((predicted == 0) & (actual == 0)).sum())
        fn = int(((predicted == 0) & (actual == 1)).sum())

        accuracy = (tp + tn) / len(scores) if len(scores) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "Metric": metric,
                "Threshold": threshold,
                "Accuracy": round(accuracy, 6),
                "Precision": round(precision, 6),
                "Recall": round(recall, 6),
                "F1": round(f1, 6),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }
        )

    return pd.DataFrame(rows)


def threshold_sweep(scores: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return pd.concat(
        [classification_metrics(scores, threshold, metrics) for threshold in THRESHOLDS],
        ignore_index=True,
    )


def best_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    ordered = sweep.sort_values(
        ["Metric", "F1", "Accuracy", "Recall", "Precision"],
        ascending=[True, False, False, False, False],
    )
    return ordered.groupby("Metric", as_index=False).head(1).reset_index(drop=True)


def evaluate_scores(
    scores: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = scores.copy()
    fixed = classification_metrics(labeled, FIXED_THRESHOLD, metrics)
    sweep = threshold_sweep(labeled, metrics)
    best = best_thresholds(sweep)
    return labeled, fixed, sweep, best


def write_evaluation_outputs(
    output_dir: Path,
    labeled: pd.DataFrame,
    fixed: pd.DataFrame,
    sweep: pd.DataFrame,
    best: pd.DataFrame,
    metrics: list[str],
    interactive_filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = output_dir / "normalized_scores.xlsx"
    labeled.to_excel(normalized, index=False)
    labeled.to_csv(output_dir / "labeled_scores.csv", index=False, encoding="utf-8-sig")
    fixed.to_csv(
        output_dir / "classification_metrics_threshold_0_50.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    best.to_csv(output_dir / "best_threshold_metrics.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_dir / "evaluation_results.xlsx", engine="openpyxl") as writer:
        fixed.to_excel(writer, sheet_name="threshold_0_50", index=False)
        sweep.to_excel(writer, sheet_name="threshold_sweep", index=False)
        best.to_excel(writer, sheet_name="best_thresholds", index=False)
        labeled.to_excel(writer, sheet_name="labeled_scores", index=False)

    build_workbook(
        input_file=normalized,
        output_file=output_dir / interactive_filename,
        metrics_arg=",".join(metrics),
    )


def model_frame(base: pd.DataFrame, scores: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    merged = base.merge(
        scores[["pair_id", config.score_column]],
        on="pair_id",
        how="left",
        validate="one_to_one",
    )
    if merged[config.score_column].isna().any():
        raise ValueError(f"{config.model_name} is missing scores after the pair_id join.")
    merged = merged.drop(columns=["pair_id"])
    merged.insert(0, "model", config.model_name)
    return merged


def summary_row(
    config: ModelConfig,
    fixed: pd.DataFrame,
    best: pd.DataFrame,
) -> dict[str, object]:
    f = fixed.iloc[0].to_dict()
    b = best.iloc[0].to_dict()
    return {
        "Model": config.model_name,
        "Metric": config.score_column,
        "Threshold_0_50_Accuracy": f["Accuracy"],
        "Threshold_0_50_Precision": f["Precision"],
        "Threshold_0_50_Recall": f["Recall"],
        "Threshold_0_50_F1": f["F1"],
        "Best_Threshold": b["Threshold"],
        "Best_Accuracy": b["Accuracy"],
        "Best_Precision": b["Precision"],
        "Best_Recall": b["Recall"],
        "Best_F1": b["F1"],
        "Output_Folder": str(config.output_dir.relative_to(WORK_ROOT)),
    }


def main() -> None:
    base = load_base_testcases()
    scores = load_and_validate_scores(base)
    summary_rows: list[dict[str, object]] = []

    for config in MODELS:
        frame = model_frame(base, scores, config)
        labeled, fixed, sweep, best = evaluate_scores(frame, [config.score_column])
        write_evaluation_outputs(
            config.output_dir,
            labeled,
            fixed,
            sweep,
            best,
            [config.score_column],
            f"interactive_threshold_{config.slug}.xlsx",
        )
        summary_rows.append(summary_row(config, fixed, best))
        print(f"Wrote {config.model_name} evaluation to {config.output_dir}")

    combined = base.drop(columns=["pair_id"]).copy()
    for config in MODELS:
        combined[config.score_column] = scores.set_index("pair_id").loc[
            base["pair_id"], config.score_column
        ].to_numpy()
    metrics = [config.score_column for config in MODELS]
    labeled, fixed, sweep, best = evaluate_scores(combined, metrics)
    write_evaluation_outputs(
        COMBINED_OUTPUT_DIR,
        labeled,
        fixed,
        sweep,
        best,
        metrics,
        "interactive_threshold_combined_llm_models.xlsx",
    )
    print(f"Wrote combined paid-LLM evaluation to {COMBINED_OUTPUT_DIR}")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_ROOT / "llm_model_evaluation_summary"
    summary.to_csv(summary_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(summary_path.with_suffix(".xlsx"), engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
