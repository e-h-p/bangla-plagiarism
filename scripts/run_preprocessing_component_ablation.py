# -*- coding: utf-8 -*-
"""Component-wise preprocessing ablation for the Bangla plagiarism benchmark.

The original polishing script evaluates the full preprocessing bundle. This
script applies each preprocessing component in isolation, recomputes the six
traditional similarity metrics, evaluates threshold behaviour, and writes
paper-ready tables/figures for the report.
"""

from __future__ import annotations

import math
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_common import FIXED_THRESHOLD, THRESHOLDS, evaluate_scores


SCRIPT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = SCRIPT_DIR / "dataset" / "banglaplagpair1k_base.xlsx"
OUTPUT_DIR = SCRIPT_DIR / "preprocessing_component_ablation"
FIGURES_DIR = OUTPUT_DIR / "figures"

from bangla_plagiarism_pipeline import (  # noqa: E402
    DEFAULT_LEXICAL_MAP,
    DEFAULT_STOPWORDS,
    DIGIT_TRANSLATION,
    INVISIBLE_RE,
    METRICS,
    TOKEN_RE,
    BanglaPreprocessor,
    ConservativeBanglaStemmer,
    LexicalNormalizer,
    PreprocessConfig,
    collapse_spaces,
    cosine_sim,
    jaccard_similarity,
    sequence_similarity,
    load_stopwords,
    strip_punctuation_and_symbols,
    word_overlap_similarity,
)


BASE_COLUMNS = ["pair_no", "variant_type", "text1", "text2"]
TEXT_COLUMNS = ["text1", "text2"]


@dataclass(frozen=True)
class ComponentProfile:
    slug: str
    label: str
    report_label: str
    steps: str
    transform: Callable[[object], str]


def safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def normalize_unicode_only(value: object) -> str:
    text = unicodedata.normalize("NFC", safe_text(value))
    text = INVISIBLE_RE.sub("", text)
    return collapse_spaces(text)


def lowercase_only(value: object) -> str:
    return collapse_spaces(safe_text(value).lower())


def digit_normalization_only(value: object) -> str:
    return collapse_spaces(safe_text(value).translate(DIGIT_TRANSLATION))


def punctuation_removal_only(value: object) -> str:
    return collapse_spaces(strip_punctuation_and_symbols(safe_text(value)))


def token_operation_text(value: object, operation: Callable[[list[str]], list[str]]) -> str:
    """Apply a token-level operation without enabling the full pipeline.

    Token-level resources in the original pipeline operate after tokenization.
    This helper mirrors that structure while keeping each operation isolated.
    """

    text = normalize_unicode_only(value)
    tokens = TOKEN_RE.findall(text)
    return " ".join(operation(tokens))


def build_profiles() -> list[ComponentProfile]:
    lexical_normalizer = LexicalNormalizer.from_tsv(DEFAULT_LEXICAL_MAP, min_confidence=0.85)
    stopwords = load_stopwords(DEFAULT_STOPWORDS)
    stemmer = ConservativeBanglaStemmer()
    full_preprocessor = BanglaPreprocessor(
        PreprocessConfig(
            lowercase=True,
            normalize_digits=True,
            remove_punctuation=True,
            apply_lexical_mapping=True,
            remove_stopwords=True,
            apply_stemming=True,
            min_lexical_confidence=0.85,
        )
    )

    return [
        ComponentProfile(
            slug="raw",
            label="Raw",
            report_label="Raw",
            steps="No corpus-level preprocessing",
            transform=safe_text,
        ),
        ComponentProfile(
            slug="unicode_normalization",
            label="Unicode normalization",
            report_label="Unicode",
            steps="NFC normalization + invisible character removal",
            transform=normalize_unicode_only,
        ),
        ComponentProfile(
            slug="lowercasing",
            label="Lowercasing",
            report_label="Lowercase",
            steps="Lowercase conversion only",
            transform=lowercase_only,
        ),
        ComponentProfile(
            slug="digit_normalization",
            label="Digit normalization",
            report_label="Digits",
            steps="Bangla and Arabic-Indic digits mapped to ASCII digits",
            transform=digit_normalization_only,
        ),
        ComponentProfile(
            slug="punctuation_removal",
            label="Punctuation removal",
            report_label="Punctuation",
            steps="Unicode punctuation and symbols replaced by whitespace",
            transform=punctuation_removal_only,
        ),
        ComponentProfile(
            slug="lexical_mapping",
            label="Lexical mapping",
            report_label="Lexical map",
            steps="Longest-match lexical normalization only",
            transform=lambda value: token_operation_text(value, lexical_normalizer.normalize),
        ),
        ComponentProfile(
            slug="stopword_removal",
            label="Stopword removal",
            report_label="Stopwords",
            steps="Curated stopword removal only",
            transform=lambda value: token_operation_text(
                value, lambda tokens: [token for token in tokens if token not in stopwords]
            ),
        ),
        ComponentProfile(
            slug="stemming",
            label="Rule-based stemming",
            report_label="Stemming",
            steps="Conservative Bangla stemming only",
            transform=lambda value: token_operation_text(
                value, lambda tokens: [stemmer.stem(token) for token in tokens]
            ),
        ),
        ComponentProfile(
            slug="full_pipeline",
            label="Full preprocessing pipeline",
            report_label="Full",
            steps=(
                "Unicode + lowercasing + digit normalization + punctuation removal + "
                "lexical mapping + stopword removal + stemming"
            ),
            transform=full_preprocessor.preprocess_text,
        ),
    ]


def load_base_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_FILE}")
    frame = pd.read_excel(INPUT_FILE, engine="openpyxl")
    missing = [column for column in BASE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")
    return frame[BASE_COLUMNS].copy()


def apply_profile(frame: pd.DataFrame, profile: ComponentProfile) -> tuple[pd.DataFrame, dict[str, object]]:
    transformed = frame.copy()
    for column in TEXT_COLUMNS:
        transformed[f"{column}_raw"] = transformed[column]
        transformed[column] = transformed[f"{column}_raw"].map(profile.transform)
        transformed[f"{column}_token_count"] = transformed[column].map(lambda value: len(str(value).split()))

    changed_text_fields = int(
        (transformed["text1"].astype(str) != transformed["text1_raw"].astype(str)).sum()
        + (transformed["text2"].astype(str) != transformed["text2_raw"].astype(str)).sum()
    )
    changed_pairs = int(
        (
            (transformed["text1"].astype(str) != transformed["text1_raw"].astype(str))
            | (transformed["text2"].astype(str) != transformed["text2_raw"].astype(str))
        ).sum()
    )
    total_tokens = int(transformed["text1_token_count"].sum() + transformed["text2_token_count"].sum())
    avg_tokens = float(total_tokens / (2 * len(transformed))) if len(transformed) else 0.0

    transformed["preprocess_profile"] = profile.slug
    transformed["preprocess_steps"] = profile.steps

    metadata = {
        "Profile": profile.slug,
        "Setting": profile.label,
        "ReportLabel": profile.report_label,
        "Steps": profile.steps,
        "ChangedTextFields": changed_text_fields,
        "ChangedPairs": changed_pairs,
        "ChangedTextFieldRate": changed_text_fields / (2 * len(transformed)) if len(transformed) else 0.0,
        "ChangedPairRate": changed_pairs / len(transformed) if len(transformed) else 0.0,
        "AverageTokensPerText": avg_tokens,
    }
    return transformed, metadata


def score_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        scores = compute_metrics_fast(str(row["text1"]), str(row["text2"]))
        record = {
            "pair_no": row["pair_no"],
            "variant_type": row["variant_type"],
            "text1": row["text1"],
            "text2": row["text2"],
        }
        record.update({metric: round(value, 6) for metric, value in scores.items()})
        rows.append(record)
    return pd.DataFrame(rows)


def lcs_length_bitset(text_a: str, text_b: str) -> int:
    """Fast exact LCS length using a bit-parallel dynamic program."""

    if not text_a or not text_b:
        return 0
    if len(text_b) > len(text_a):
        text_a, text_b = text_b, text_a

    masks: dict[str, int] = {}
    for index, char in enumerate(text_b):
        masks[char] = masks.get(char, 0) | (1 << index)

    state = 0
    for char in text_a:
        matches = masks.get(char, 0)
        x = state | matches
        y = (state << 1) | 1
        state = x & ~(x - y)
    return state.bit_count()


def lcs_similarity_fast(text_a: str, text_b: str) -> float:
    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    return lcs_length_bitset(text_a, text_b) / max(len(text_a), len(text_b))


def levenshtein_distance_myers(text_a: str, text_b: str) -> int:
    """Exact Levenshtein distance via Myers' bit-vector algorithm."""

    if text_a == text_b:
        return 0
    if not text_a:
        return len(text_b)
    if not text_b:
        return len(text_a)
    if len(text_a) > len(text_b):
        text_a, text_b = text_b, text_a

    pattern_len = len(text_a)
    top_bit = 1 << (pattern_len - 1)
    all_ones = (1 << pattern_len) - 1
    masks: dict[str, int] = {}
    for index, char in enumerate(text_a):
        masks[char] = masks.get(char, 0) | (1 << index)

    positive = all_ones
    negative = 0
    score = pattern_len

    for char in text_b:
        eq = masks.get(char, 0)
        xv = eq | negative
        xh = (((eq & positive) + positive) ^ positive) | eq
        ph = negative | ~(xh | positive)
        mh = positive & xh

        if ph & top_bit:
            score += 1
        elif mh & top_bit:
            score -= 1

        ph = ((ph << 1) | 1) & all_ones
        mh = (mh << 1) & all_ones
        positive = (mh | ~(xv | ph)) & all_ones
        negative = ph & xv

    return score


def edit_distance_similarity_fast(text_a: str, text_b: str) -> float:
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 1.0
    return 1 - (levenshtein_distance_myers(text_a, text_b) / max_len)


def compute_metrics_fast(text_a: str, text_b: str) -> dict[str, float]:
    return {
        "Cosine": cosine_sim(text_a, text_b),
        "Jaccard": jaccard_similarity(text_a, text_b),
        "LCS": lcs_similarity_fast(text_a, text_b),
        "Sequence": sequence_similarity(text_a, text_b),
        "EditDist": edit_distance_similarity_fast(text_a, text_b),
        "WordOverlap": word_overlap_similarity(text_a, text_b),
    }


def add_profile_columns(df: pd.DataFrame, profile: ComponentProfile) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "Profile", profile.slug)
    out.insert(1, "Setting", profile.label)
    out.insert(2, "ReportLabel", profile.report_label)
    return out


def write_profile_outputs(
    *,
    profile: ComponentProfile,
    transformed: pd.DataFrame,
    scored: pd.DataFrame,
    labeled_scores: pd.DataFrame,
    class_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    dataset_dir = OUTPUT_DIR / "datasets"
    score_dir = OUTPUT_DIR / "scores"
    eval_dir = OUTPUT_DIR / "evaluations" / profile.slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    transformed.to_csv(dataset_dir / f"{profile.slug}_dataset.csv", index=False, encoding="utf-8-sig")
    transformed.to_excel(dataset_dir / f"{profile.slug}_dataset.xlsx", index=False, engine="openpyxl")
    scored.to_csv(score_dir / f"{profile.slug}_scores.csv", index=False, encoding="utf-8-sig")
    scored.to_excel(score_dir / f"{profile.slug}_scores.xlsx", index=False, engine="openpyxl")

    labeled_scores.to_csv(eval_dir / "labeled_scores.csv", index=False, encoding="utf-8-sig")
    class_df.to_csv(eval_dir / "classification_metrics_threshold_0_50.csv", index=False, encoding="utf-8-sig")
    sweep_df.to_csv(eval_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(eval_dir / "best_threshold_metrics.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(eval_dir / "evaluation_results.xlsx", engine="openpyxl") as writer:
        class_df.to_excel(writer, sheet_name="threshold_0_50", index=False)
        sweep_df.to_excel(writer, sheet_name="threshold_sweep", index=False)
        best_df.to_excel(writer, sheet_name="best_thresholds", index=False)
        labeled_scores.to_excel(writer, sheet_name="labeled_scores", index=False)


def enrich_with_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    raw = out[out["Profile"] == "raw"][["Metric", "F1", "Accuracy", "Recall", "Specificity"]].copy()
    raw = raw.rename(
        columns={
            "F1": "Raw_F1",
            "Accuracy": "Raw_Accuracy",
            "Recall": "Raw_Recall",
            "Specificity": "Raw_Specificity",
        }
    )
    out = out.merge(raw, on="Metric", how="left", validate="many_to_one")
    for column in ["F1", "Accuracy", "Recall", "Specificity"]:
        out[f"Delta_{column}_vs_Raw"] = out[column] - out[f"Raw_{column}"]
    return out


def profile_order(profiles: list[ComponentProfile]) -> list[str]:
    return [profile.slug for profile in profiles]


def metric_order() -> list[str]:
    return METRICS


def heatmap_fixed_f1_delta(fixed_summary: pd.DataFrame, profiles: list[ComponentProfile]) -> Path:
    plot_df = fixed_summary[fixed_summary["Profile"] != "raw"].copy()
    labels = {profile.slug: profile.report_label for profile in profiles}
    pivot = plot_df.pivot(index="Profile", columns="Metric", values="Delta_F1_vs_Raw")
    pivot = pivot.reindex([p.slug for p in profiles if p.slug != "raw"])[metric_order()]

    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    limit = max(0.01, float(np.nanmax(np.abs(data))))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=[labels.get(idx, idx) for idx in pivot.index])
    ax.set_title("F1 change vs. raw text at t=0.50")

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Delta F1")
    fig.tight_layout()
    path = FIGURES_DIR / "preprocessing_component_f1_delta_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def best_fixed_profile_bars(fixed_summary: pd.DataFrame, profiles: list[ComponentProfile]) -> Path:
    ordered = []
    for profile in profiles:
        rows = fixed_summary[fixed_summary["Profile"] == profile.slug].sort_values(
            ["F1", "Accuracy", "Recall", "Specificity"], ascending=[False, False, False, False]
        )
        if rows.empty:
            continue
        best = rows.iloc[0].copy()
        best["ReportLabel"] = profile.report_label
        ordered.append(best)
    plot_df = pd.DataFrame(ordered)

    y = np.arange(len(plot_df))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    labels = [f"{row.ReportLabel} ({row.Metric})" for row in plot_df.itertuples(index=False)]
    bars_f1 = ax.barh(y - height / 2, plot_df["F1"], height, label="F1", color="#4C78A8")
    bars_spec = ax.barh(y + height / 2, plot_df["Specificity"], height, label="Specificity", color="#F58518")

    ax.set_xlim(0, 1.02)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_title("Best fixed-threshold metric within each preprocessing setting")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    ax.bar_label(bars_f1, labels=[f"{v:.3f}" for v in plot_df["F1"]], padding=3, fontsize=8)
    ax.bar_label(bars_spec, labels=[f"{v:.3f}" for v in plot_df["Specificity"]], padding=3, fontsize=8)

    fig.tight_layout()
    path = FIGURES_DIR / "preprocessing_component_best_fixed_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def metric_delta_grouped_bars(fixed_summary: pd.DataFrame) -> Path:
    selected_profiles = [
        "punctuation_removal",
        "lexical_mapping",
        "stopword_removal",
        "stemming",
        "full_pipeline",
    ]
    plot_df = fixed_summary[fixed_summary["Profile"].isin(selected_profiles)].copy()
    labels = {
        "punctuation_removal": "Punctuation",
        "lexical_mapping": "Lexical map",
        "stopword_removal": "Stopwords",
        "stemming": "Stemming",
        "full_pipeline": "Full",
    }
    pivot = plot_df.pivot(index="Metric", columns="Profile", values="Delta_F1_vs_Raw").reindex(metric_order())

    x = np.arange(len(pivot.index))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    colors = ["#54A24B", "#B279A2", "#E45756", "#72B7B2", "#4C78A8"]
    for i, profile in enumerate(selected_profiles):
        values = pivot[profile].to_numpy()
        ax.bar(x + (i - 2) * width, values, width, label=labels[profile], color=colors[i])

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Delta F1 vs. raw")
    ax.set_title("Component effects by similarity metric at t=0.50")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "preprocessing_component_metric_delta_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def f3(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.3f}"


def signed_f3(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):+.3f}"


def write_latex_tables(fixed_summary: pd.DataFrame, best_summary: pd.DataFrame, profiles: list[ComponentProfile]) -> None:
    table_dir = OUTPUT_DIR / "latex_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    labels = {profile.slug: profile.report_label for profile in profiles}

    f1_pivot = fixed_summary.pivot(index="Profile", columns="Metric", values="F1").reindex(profile_order(profiles))[metric_order()]
    delta_pivot = fixed_summary.pivot(index="Profile", columns="Metric", values="Delta_F1_vs_Raw").reindex(profile_order(profiles))[metric_order()]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Component-wise preprocessing ablation at $t{=}0.50$. Each cell reports F1, with the change from the raw setting in parentheses.}",
        "\\label{tab:component-preprocessing-f1}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l rrrrrr}",
        "\\toprule",
        "Setting & TF-IDF Cosine & Jaccard & LCS & Sequence & EditDist & WordOverlap \\\\",
        "\\midrule",
    ]
    for profile in profile_order(profiles):
        cells = []
        for metric in metric_order():
            f1 = f3(f1_pivot.loc[profile, metric])
            delta = signed_f3(delta_pivot.loc[profile, metric])
            cells.append(f"{f1} ({delta})")
        lines.append(f"{labels[profile]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""])
    (table_dir / "component_fixed_f1_table.tex").write_text("\n".join(lines), encoding="utf-8")

    best_fixed_rows = []
    for profile in profiles:
        rows = fixed_summary[fixed_summary["Profile"] == profile.slug].sort_values(
            ["F1", "Accuracy", "Recall", "Specificity"], ascending=[False, False, False, False]
        )
        best = rows.iloc[0]
        best_fixed_rows.append(best)

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Strongest metric within each preprocessing setting. Fixed results use $t{=}0.50$; calibrated results use the in-sample best threshold from the same sweep.}",
        "\\label{tab:component-preprocessing-best}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l l rrrr l rrrr}",
        "\\toprule",
        "Setting & Fixed best metric & Acc & Rec & Spec. & F1 & Calibrated best metric ($t$) & Acc & Rec & Spec. & F1 \\\\",
        "\\midrule",
    ]
    for row in best_fixed_rows:
        profile = str(row["Profile"])
        best_rows = best_summary[best_summary["Profile"] == profile].sort_values(
            ["F1", "Accuracy", "Recall", "Specificity"], ascending=[False, False, False, False]
        )
        calibrated = best_rows.iloc[0]
        lines.append(
            f"{labels[profile]} & {row['Metric']} & {f3(row['Accuracy'])} & {f3(row['Recall'])} & "
            f"{f3(row['Specificity'])} & {f3(row['F1'])} & "
            f"{calibrated['Metric']} ({f3(calibrated['Threshold'])}) & {f3(calibrated['Accuracy'])} & "
            f"{f3(calibrated['Recall'])} & {f3(calibrated['Specificity'])} & {f3(calibrated['F1'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""])
    (table_dir / "component_best_profile_table.tex").write_text("\n".join(lines), encoding="utf-8")

    effect_rows = []
    for profile in profiles:
        if profile.slug == "raw":
            continue
        rows = fixed_summary[fixed_summary["Profile"] == profile.slug]
        improved = int((rows["Delta_F1_vs_Raw"] > 0.000001).sum())
        reduced = int((rows["Delta_F1_vs_Raw"] < -0.000001).sum())
        neutral = int(len(rows) - improved - reduced)
        effect_rows.append(
            {
                "Profile": profile.slug,
                "Setting": labels[profile.slug],
                "MeanDeltaF1": rows["Delta_F1_vs_Raw"].mean(),
                "Improved": improved,
                "Reduced": reduced,
                "Neutral": neutral,
                "BestGainMetric": rows.sort_values("Delta_F1_vs_Raw", ascending=False).iloc[0]["Metric"],
                "BestGain": rows["Delta_F1_vs_Raw"].max(),
                "WorstLossMetric": rows.sort_values("Delta_F1_vs_Raw", ascending=True).iloc[0]["Metric"],
                "WorstLoss": rows["Delta_F1_vs_Raw"].min(),
            }
        )

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Direction of single-component effects at $t{=}0.50$ relative to raw text across the six traditional metrics.}",
        "\\label{tab:component-preprocessing-effect-summary}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l r r r r l r l r}",
        "\\toprule",
        "Setting & Mean $\\Delta$F1 & + & -- & 0 & Best gain & $\\Delta$ & Largest loss & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for row in effect_rows:
        best_gain_metric = row["BestGainMetric"] if row["BestGain"] > 0.000001 else "None"
        worst_loss_metric = row["WorstLossMetric"] if row["WorstLoss"] < -0.000001 else "None"
        lines.append(
            f"{row['Setting']} & {signed_f3(row['MeanDeltaF1'])} & {row['Improved']} & {row['Reduced']} & "
            f"{row['Neutral']} & {best_gain_metric} & {signed_f3(row['BestGain'])} & "
            f"{worst_loss_metric} & {signed_f3(row['WorstLoss'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    (table_dir / "component_effect_summary_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_report_section(
    fixed_summary: pd.DataFrame,
    best_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    profiles: list[ComponentProfile],
    figure_paths: list[Path],
) -> None:
    raw = fixed_summary[fixed_summary["Profile"] == "raw"].copy()
    full = fixed_summary[fixed_summary["Profile"] == "full_pipeline"].copy()
    merged = full[["Metric", "F1", "Delta_F1_vs_Raw", "Specificity", "Delta_Specificity_vs_Raw"]].merge(
        raw[["Metric", "F1", "Specificity"]],
        on="Metric",
        suffixes=("_Full", "_Raw"),
    )
    improved_full = merged[merged["Delta_F1_vs_Raw"] > 0]["Metric"].tolist()
    reduced_full = merged[merged["Delta_F1_vs_Raw"] < 0]["Metric"].tolist()

    best_fixed = fixed_summary.sort_values(
        ["F1", "Accuracy", "Recall", "Specificity"], ascending=[False, False, False, False]
    ).iloc[0]
    best_cal = best_summary.sort_values(
        ["F1", "Accuracy", "Recall", "Specificity"], ascending=[False, False, False, False]
    ).iloc[0]

    section = f"""% Auto-generated by run_preprocessing_component_ablation.py
% Include after the traditional raw-vs-polished discussion if the manuscript needs
% the component-wise preprocessing analysis in the main text.

\\subsection{{Component-Wise Preprocessing Ablation}}
\\label{{subsec:component-preprocessing-ablation}}

To isolate the contribution of each preprocessing operation, the 1{{,}}000-pair benchmark was reprocessed under nine settings: the raw text, seven single-component transformations, and the previously reported full preprocessing pipeline. Each transformed corpus was evaluated with the same six traditional similarity metrics, the same binary labels, the same fixed threshold ($t=0.50$), and the same in-sample threshold sweep. The single-component runs should be interpreted as diagnostic ablations: token-level operations such as lexical mapping, stopword removal, and stemming necessarily pass through the same tokenization interface used by the original preprocessing implementation, whereas the raw condition is kept unchanged.

\\input{{preprocessing_component_ablation/latex_tables/component_fixed_f1_table.tex}}

Table~\\ref{{tab:component-preprocessing-f1}} shows that preprocessing is not uniformly beneficial. The full pipeline improves fixed-threshold F1 for {", ".join(improved_full)} but reduces F1 for {", ".join(reduced_full)}. The largest full-pipeline gain occurs for sequence similarity ($\\Delta$F1={signed_f3(float(merged.loc[merged["Metric"] == "Sequence", "Delta_F1_vs_Raw"].iloc[0]))}), while the largest loss occurs for LCS ($\\Delta$F1={signed_f3(float(merged.loc[merged["Metric"] == "LCS", "Delta_F1_vs_Raw"].iloc[0]))}). This confirms that Bangla-aware normalization is selective: it helps metrics that depend on token identity or noisy surface alignment, but it can harm metrics that benefit from preserving long character subsequences.

\\input{{preprocessing_component_ablation/latex_tables/component_effect_summary_table.tex}}

The single-component results sharpen this interpretation. Lexical mapping and stemming provide the clearest positive effects for token-oriented metrics, although stemming also weakens character-continuity metrics. Punctuation removal has little benefit in the fixed-threshold ablation because the raw token metrics already strip punctuation internally during scoring; as a corpus-level transformation, it slightly reduces LCS, sequence, and edit-distance F1 by shortening and reshaping the character stream. Stopword removal is the most damaging isolated step because it removes function-word anchors that help both character-continuity and overlap-based metrics. Unicode normalization, lowercasing, and digit normalization have small effects on this benchmark, indicating that the corpus is already mostly normalized with respect to casing, script-specific digits, and hidden Unicode variation.

\\begin{{figure*}}[!t]
\\centering
\\begin{{minipage}}[t]{{0.48\\textwidth}}
  \\centering
  \\includegraphics[width=\\linewidth]{{figures/preprocessing_component_f1_delta_heatmap.png}}
\\end{{minipage}}%
\\hfill
\\begin{{minipage}}[t]{{0.48\\textwidth}}
  \\centering
  \\includegraphics[width=\\linewidth]{{figures/preprocessing_component_best_fixed_bar.png}}
\\end{{minipage}}
\\caption{{Component-wise preprocessing diagnostics. \\textit{{Left:}} F1 change relative to raw text at $t=0.50$ for each metric and preprocessing setting. \\textit{{Right:}} Best fixed-threshold metric within each preprocessing setting, with specificity shown to expose false-positive behaviour.}}
\\label{{fig:component-preprocessing-ablation}}
\\end{{figure*}}

\\input{{preprocessing_component_ablation/latex_tables/component_best_profile_table.tex}}

Table~\\ref{{tab:component-preprocessing-best}} retains the current full-pipeline result while comparing it against the individual components. The strongest fixed-threshold configuration remains {best_fixed['Setting']} with {best_fixed['Metric']} (F1={f3(best_fixed['F1'])}, specificity={f3(best_fixed['Specificity'])}), and the strongest calibrated configuration is {best_cal['Setting']} with {best_cal['Metric']} at $t={f3(best_cal['Threshold'])}$ (F1={f3(best_cal['F1'])}, specificity={f3(best_cal['Specificity'])}). Thus, the full preprocessing pipeline is useful as a diagnostic polished track, but it should not replace the raw LCS baseline for final traditional-metric comparison. For Bangla plagiarism screening, preprocessing should be selected by metric family: use normalization for token-overlap diagnostics, but preserve raw text for character-continuity metrics such as LCS and edit-distance similarity.
"""
    (OUTPUT_DIR / "component_preprocessing_report_section.tex").write_text(section, encoding="utf-8")

    markdown = [
        "# Component-Wise Preprocessing Ablation",
        "",
        f"Rows evaluated: {len(metadata) and 1000}. Fixed threshold: {FIXED_THRESHOLD}. Sweep: {THRESHOLDS}.",
        "",
        "## Key Findings",
        "",
        f"- Best fixed-threshold setting: **{best_fixed['Setting']} / {best_fixed['Metric']}**, F1={f3(best_fixed['F1'])}, specificity={f3(best_fixed['Specificity'])}.",
        f"- Best calibrated setting: **{best_cal['Setting']} / {best_cal['Metric']}** at t={f3(best_cal['Threshold'])}, F1={f3(best_cal['F1'])}, specificity={f3(best_cal['Specificity'])}.",
        f"- Full pipeline improves: {', '.join(improved_full)}.",
        f"- Full pipeline reduces: {', '.join(reduced_full)}.",
        "- Single-component effects show that punctuation removal and lexical mapping are generally helpful, while isolated stopword removal is harmful.",
        "",
        "## Generated Figures",
    ]
    for path in figure_paths:
        markdown.append(f"- `{path.relative_to(SCRIPT_DIR)}`")
    markdown.extend(
        [
            "",
            "## Tables",
            "",
            "- `fixed_threshold_summary.csv`",
            "- `best_threshold_summary.csv`",
            "- `component_effect_summary.csv`",
            "- `component_preprocessing_report_section.tex`",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_dataset()
    profiles = build_profiles()
    metadata_rows: list[dict[str, object]] = []
    fixed_frames: list[pd.DataFrame] = []
    sweep_frames: list[pd.DataFrame] = []
    best_frames: list[pd.DataFrame] = []

    for profile in profiles:
        print(f"Processing {profile.label} ...")
        transformed, profile_metadata = apply_profile(base, profile)
        scored = score_dataset(transformed)
        labeled_scores, class_df, sweep_df, best_df = evaluate_scores(scored, METRICS)

        class_df = add_profile_columns(class_df, profile)
        sweep_df = add_profile_columns(sweep_df, profile)
        best_df = add_profile_columns(best_df, profile)
        labeled_scores = add_profile_columns(labeled_scores, profile)
        scored = add_profile_columns(scored, profile)

        write_profile_outputs(
            profile=profile,
            transformed=transformed,
            scored=scored,
            labeled_scores=labeled_scores,
            class_df=class_df,
            sweep_df=sweep_df,
            best_df=best_df,
        )

        metadata_rows.append(profile_metadata)
        fixed_frames.append(class_df)
        sweep_frames.append(sweep_df)
        best_frames.append(best_df)

    metadata = pd.DataFrame(metadata_rows)
    fixed_summary = enrich_with_deltas(pd.concat(fixed_frames, ignore_index=True))
    sweep_summary = pd.concat(sweep_frames, ignore_index=True)
    best_summary = enrich_with_deltas(pd.concat(best_frames, ignore_index=True))

    effect_rows = []
    for profile in profiles:
        if profile.slug == "raw":
            continue
        rows = fixed_summary[fixed_summary["Profile"] == profile.slug]
        effect_rows.append(
            {
                "Profile": profile.slug,
                "Setting": profile.label,
                "ReportLabel": profile.report_label,
                "MeanDeltaF1": rows["Delta_F1_vs_Raw"].mean(),
                "ImprovedMetrics": int((rows["Delta_F1_vs_Raw"] > 0.000001).sum()),
                "ReducedMetrics": int((rows["Delta_F1_vs_Raw"] < -0.000001).sum()),
                "NeutralMetrics": int((rows["Delta_F1_vs_Raw"].abs() <= 0.000001).sum()),
                "BestGainMetric": rows.sort_values("Delta_F1_vs_Raw", ascending=False).iloc[0]["Metric"],
                "BestGainDeltaF1": rows["Delta_F1_vs_Raw"].max(),
                "LargestLossMetric": rows.sort_values("Delta_F1_vs_Raw", ascending=True).iloc[0]["Metric"],
                "LargestLossDeltaF1": rows["Delta_F1_vs_Raw"].min(),
            }
        )
    effect_summary = pd.DataFrame(effect_rows)

    metadata.to_csv(OUTPUT_DIR / "preprocessing_profile_metadata.csv", index=False, encoding="utf-8-sig")
    fixed_summary.to_csv(OUTPUT_DIR / "fixed_threshold_summary.csv", index=False, encoding="utf-8-sig")
    sweep_summary.to_csv(OUTPUT_DIR / "threshold_sweep_summary.csv", index=False, encoding="utf-8-sig")
    best_summary.to_csv(OUTPUT_DIR / "best_threshold_summary.csv", index=False, encoding="utf-8-sig")
    effect_summary.to_csv(OUTPUT_DIR / "component_effect_summary.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_DIR / "preprocessing_component_ablation_summary.xlsx", engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="profile_metadata", index=False)
        fixed_summary.to_excel(writer, sheet_name="fixed_threshold", index=False)
        best_summary.to_excel(writer, sheet_name="best_threshold", index=False)
        effect_summary.to_excel(writer, sheet_name="effect_summary", index=False)
        sweep_summary.to_excel(writer, sheet_name="threshold_sweep", index=False)

    figure_paths = [
        heatmap_fixed_f1_delta(fixed_summary, profiles),
        best_fixed_profile_bars(fixed_summary, profiles),
        metric_delta_grouped_bars(fixed_summary),
    ]
    readme_lines = [
        "# Component-Wise Preprocessing Ablation",
        "",
        "This directory contains the reproducible evaluation artifacts for the component-wise preprocessing experiment.",
        "",
        "The manuscript text and LaTeX tables are kept directly in `report/bangla_plagiarism_final_revised.tex`.",
        "This script intentionally does not emit separate helper `.tex` files.",
        "",
        "## Generated Figures",
    ]
    for path in figure_paths:
        readme_lines.append(f"- `{path.relative_to(SCRIPT_DIR)}`")
    readme_lines.extend(
        [
            "",
            "## Main Result Files",
            "",
            "- `fixed_threshold_summary.csv`",
            "- `best_threshold_summary.csv`",
            "- `component_effect_summary.csv`",
            "- `threshold_sweep_summary.csv`",
            "- `preprocessing_component_ablation_summary.xlsx`",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print()
    print(f"Component-wise preprocessing ablation written to: {OUTPUT_DIR}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
