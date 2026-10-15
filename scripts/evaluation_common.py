from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from build_interactive_threshold_workbook import build_workbook


FIXED_THRESHOLD = 0.50
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
BASE_REQUIRED_COLUMNS = ["pair_no", "variant_type", "text1", "text2"]
OUTPUT_BASE_COLUMNS = ["pair_no", "dataset", "source_case_no", "variant_type", "text1", "text2"]


@dataclass(frozen=True)
class ScoreWorkbookConfig:
    model_name: str
    slug: str
    input_file: Path
    score_columns: tuple[str, ...]
    metric_column: str
    text1_column: str = "text1"
    text2_column: str = "text2"
    id_column: str = "pair_no"
    sheet_name: str = "Sheet1"
    detection_column: str | None = "Detection Category"
    clip_scores: bool = False


def load_base_testcases(base_path: Path) -> pd.DataFrame:
    base = pd.read_excel(base_path, engine="openpyxl")
    missing = [column for column in BASE_REQUIRED_COLUMNS if column not in base.columns]
    if missing:
        raise ValueError(f"Base testcase file is missing columns: {missing}")
    if not base["pair_no"].is_unique:
        raise ValueError("Base testcase pair_no values must be unique.")

    prepared = base.copy()
    prepared["variant_type"] = prepared["variant_type"].astype(str)
    prepared["dataset"] = prepared["variant_type"].str.lower().eq("mismatch").map({True: "mismatch", False: "textcase"})
    prepared["source_case_no"] = prepared["pair_no"]
    prepared = prepared[OUTPUT_BASE_COLUMNS].copy()
    prepared["Label"] = prepared["dataset"].astype(str).str.lower().ne("mismatch").astype(int)
    return prepared


def classification_metrics(scores: pd.DataFrame, threshold: float, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for metric in metrics:
        predicted = (scores[metric] >= threshold).astype(int)
        actual = scores["Label"].astype(int)

        tp = int(((predicted == 1) & (actual == 1)).sum())
        fp = int(((predicted == 1) & (actual == 0)).sum())
        tn = int(((predicted == 0) & (actual == 0)).sum())
        fn = int(((predicted == 0) & (actual == 1)).sum())

        accuracy = (tp + tn) / len(scores) if len(scores) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        balanced_accuracy = (recall + specificity) / 2
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "Metric": metric,
                "Threshold": threshold,
                "Accuracy": round(accuracy, 6),
                "Precision": round(precision, 6),
                "Recall": round(recall, 6),
                "Specificity": round(specificity, 6),
                "FPR": round(fpr, 6),
                "BalancedAccuracy": round(balanced_accuracy, 6),
                "F1": round(f1, 6),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }
        )

    return pd.DataFrame(rows)


def threshold_sweep(scores: pd.DataFrame, thresholds: list[float], metrics: list[str]) -> pd.DataFrame:
    frames = [classification_metrics(scores, threshold, metrics) for threshold in thresholds]
    return pd.concat(frames, ignore_index=True)


def best_thresholds(sweep_df: pd.DataFrame) -> pd.DataFrame:
    ordered = sweep_df.sort_values(
        ["Metric", "F1", "Accuracy", "Recall", "Precision", "Threshold"],
        ascending=[True, False, False, False, False, True],
    )
    return ordered.groupby("Metric", sort=False).first().reset_index()


def evaluate_scores(scores: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled_scores = scores.copy()
    if "Label" not in labeled_scores.columns:
        if "dataset" in labeled_scores.columns:
            labeled_scores["Label"] = labeled_scores["dataset"].astype(str).str.lower().ne("mismatch").astype(int)
        elif "variant_type" in labeled_scores.columns:
            labeled_scores["Label"] = labeled_scores["variant_type"].astype(str).str.lower().ne("mismatch").astype(int)
        else:
            raise ValueError("Unable to derive Label column.")

    class_df = classification_metrics(labeled_scores, FIXED_THRESHOLD, metrics)
    sweep_df = threshold_sweep(labeled_scores, THRESHOLDS, metrics)
    best_df = best_thresholds(sweep_df)
    return labeled_scores, class_df, sweep_df, best_df


def select_score_series(df: pd.DataFrame, config: ScoreWorkbookConfig) -> tuple[pd.Series, list[str]]:
    available_columns = [column for column in config.score_columns if column in df.columns]
    if not available_columns:
        raise ValueError(f"{config.model_name} file is missing all score columns: {list(config.score_columns)}")

    numeric_parts = [
        pd.to_numeric(df[column], errors="coerce").rename(column)
        for column in available_columns
    ]
    combined = pd.concat(numeric_parts, axis=1).bfill(axis=1).iloc[:, 0]
    return combined, available_columns


def normalize_model_scores(base: pd.DataFrame, config: ScoreWorkbookConfig) -> tuple[pd.DataFrame, int, list[str]]:
    df = pd.read_excel(config.input_file, sheet_name=config.sheet_name, engine="openpyxl")
    required_columns = [config.id_column, config.text1_column, config.text2_column]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{config.model_name} file is missing columns: {missing}")
    if not df[config.id_column].is_unique:
        raise ValueError(f"{config.model_name} id column contains duplicate values: {config.id_column}")

    score_series, used_score_columns = select_score_series(df, config)
    filled_missing = int(score_series.isna().sum())
    if filled_missing or not np.isfinite(score_series.to_numpy(dtype=float)).all():
        raise ValueError(f"{config.model_name}: scores must be finite and complete; no imputation is permitted")
    if not config.clip_scores and not score_series.between(0.0, 1.0).all():
        raise ValueError(f"{config.model_name}: score outside [0,1]")
    evaluated_scores = score_series.clip(0.0, 1.0) if config.clip_scores else score_series

    score_frame = pd.DataFrame(
        {
            "pair_no": df[config.id_column],
            config.metric_column: evaluated_scores,
        }
    )

    if config.detection_column and config.detection_column in df.columns:
        score_frame["model_detection_category"] = df[config.detection_column]

    merged = base.merge(score_frame, on="pair_no", how="left", validate="one_to_one")
    if merged[config.metric_column].isna().any():
        missing_ids = merged.loc[merged[config.metric_column].isna(), "pair_no"].tolist()
        raise ValueError(f"{config.model_name} has no score for pair_no values: {missing_ids[:10]}")

    sorted_model = df.sort_values(config.id_column).reset_index(drop=True)
    sorted_base = base.sort_values("pair_no").reset_index(drop=True)
    text1_matches = sorted_model[config.text1_column].astype(str).equals(sorted_base["text1"].astype(str))
    text2_matches = sorted_model[config.text2_column].astype(str).equals(sorted_base["text2"].astype(str))
    if not (text1_matches and text2_matches):
        raise ValueError(f"{config.model_name} text columns do not match the base testcase file.")

    merged.insert(0, "model", config.model_name)
    return merged, filled_missing, used_score_columns


def build_combined_scores(base: pd.DataFrame, configs: list[ScoreWorkbookConfig], model_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = base.copy()
    for config in configs:
        scores = model_scores[config.slug][["pair_no", config.metric_column]].copy()
        combined = combined.merge(scores, on="pair_no", how="left", validate="one_to_one")
    return combined


def write_evaluation_outputs(
    output_dir: Path,
    labeled_scores: pd.DataFrame,
    class_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    best_df: pd.DataFrame,
    metrics: list[str],
    interactive_filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_input = output_dir / "normalized_scores.xlsx"
    labeled_scores.to_excel(normalized_input, index=False)
    labeled_scores.to_csv(output_dir / "labeled_scores.csv", index=False, encoding="utf-8-sig")
    class_df.to_csv(output_dir / "classification_metrics_threshold_0_50.csv", index=False, encoding="utf-8-sig")
    sweep_df.to_csv(output_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(output_dir / "best_threshold_metrics.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_dir / "evaluation_results.xlsx", engine="openpyxl") as writer:
        class_df.to_excel(writer, sheet_name="threshold_0_50", index=False)
        sweep_df.to_excel(writer, sheet_name="threshold_sweep", index=False)
        best_df.to_excel(writer, sheet_name="best_thresholds", index=False)
        labeled_scores.to_excel(writer, sheet_name="labeled_scores", index=False)

    build_workbook(
        input_file=normalized_input,
        output_file=output_dir / interactive_filename,
        metrics_arg=",".join(metrics),
    )


def summary_row(config: ScoreWorkbookConfig, class_df: pd.DataFrame, best_df: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    fixed = class_df.iloc[0].to_dict()
    best = best_df.iloc[0].to_dict()
    return {
        "Model": config.model_name,
        "Metric": config.metric_column,
        "Threshold_0_50_Accuracy": fixed["Accuracy"],
        "Threshold_0_50_Precision": fixed["Precision"],
        "Threshold_0_50_Recall": fixed["Recall"],
        "Threshold_0_50_Specificity": fixed["Specificity"],
        "Threshold_0_50_FPR": fixed["FPR"],
        "Threshold_0_50_BalancedAccuracy": fixed["BalancedAccuracy"],
        "Threshold_0_50_F1": fixed["F1"],
        "Best_Threshold": best["Threshold"],
        "Best_Accuracy": best["Accuracy"],
        "Best_Precision": best["Precision"],
        "Best_Recall": best["Recall"],
        "Best_Specificity": best["Specificity"],
        "Best_FPR": best["FPR"],
        "Best_BalancedAccuracy": best["BalancedAccuracy"],
        "Best_F1": best["F1"],
        "Output_Folder": str(output_dir),
    }

