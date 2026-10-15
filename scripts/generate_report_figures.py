# -*- coding: utf-8 -*-
"""Generate every figure used by the updated BanglaPlagPair-1K report.

Reads only the evaluation artefacts produced by the experiment suite, so the
figures cannot drift from the tables.

    python generate_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WORK_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = WORK_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"

RESULTS = WORK_ROOT / "results"
ABLATION = WORK_ROOT / "preprocessing_component_ablation"
SENT = WORK_ROOT / "Sentence embedding models evualtion"
FREE = WORK_ROOT / "free llm evualation"
PAID = WORK_ROOT / "paid llm evualation"
PROMPT = WORK_ROOT / "prompt_engineering_evaluation"
BBERT = WORK_ROOT / "banglabert_evaluation"

TRADITIONAL = ["Cosine", "Jaccard", "LCS", "Sequence", "EditDist", "WordOverlap"]
SENT_METRICS = ["LaBSE_Score", "E5_Base_Score", "BGE_M3_Score", "MiniLM_Multilingual_Score"]
SENT_LABEL = {"LaBSE_Score": "LaBSE", "E5_Base_Score": "E5-base",
              "BGE_M3_Score": "BGE-M3", "MiniLM_Multilingual_Score": "MiniLM"}

PROMPT_SHORT = {"P1": "Few-shot", "P2": "Contradict.", "P3": "Hard-neg.",
                "P4": "Bangla instr.", "P5": "NLI", "P6": "Multi-dim.",
                "P7": "Calib.", "P8": "Direct"}
PROMPT_TICK = {"P1": "P1\nFew-shot", "P2": "P2\nContrad.",
               "P3": "P3\nHard neg.", "P4": "P4\nBangla\ninstr.",
               "P5": "P5\nNLI", "P6": "P6\nMulti-dim.",
               "P7": "P7\nCalib.", "P8": "P8\nDirect"}
MODEL_COLOR = {"Gemma 2 9B IT": "#2563eb",
               "BanglaLLama 3 8B Instruct": "#16a34a",
               "Llama 3 8B Bangla 4bit": "#ea580c"}
MODEL_SHORT = {"Gemma 2 9B IT": "Gemma 2 9B IT",
               "BanglaLLama 3 8B Instruct": "BanglaLLama 3 8B",
               "Llama 3 8B Bangla 4bit": "Llama 3 8B Bangla"}
PAID_LABEL = {
    "DeepSeek_V4_Flash_Score": "DeepSeek V4 Flash",
    "GPT_5_6_Luna_Score": "GPT 5.6 Luna",
    "Gemini_3_5_Flash_Lite_Score": "Gemini 3.5 Flash Lite",
    "Qwen_3_7_Flash_Score": "Qwen 3.7 Flash",
    "Claude_Sonnet_5_Score": "Claude Sonnet 5",
    "Gemini_3_6_Flash_Score": "Gemini 3.6 Flash",
    "DeepSeek_V4_Pro_0813_Score": "DeepSeek V4 Pro (0813)",
    "Grok_4_3_Score": "Grok 4.3",
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.family": "DejaVu Sans",
        "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
        "axes.edgecolor": "#374151", "grid.color": "#d1d5db", "grid.linewidth": 0.55,
    })


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=420, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def confusion(scores: pd.Series, labels: pd.Series, t: float) -> dict:
    pred = (scores >= t).astype(int)
    act = labels.astype(int)
    tp = int(((pred == 1) & (act == 1)).sum()); fp = int(((pred == 1) & (act == 0)).sum())
    tn = int(((pred == 0) & (act == 0)).sum()); fn = int(((pred == 0) & (act == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision": prec, "Recall": rec, "Specificity": spec,
            "Accuracy": (tp + tn) / len(scores),
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def with_specificity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add specificity when reading the historical paid-LLM table schema."""
    frame = frame.copy()
    if "Specificity" not in frame.columns:
        denominator = frame["TN"] + frame["FP"]
        frame["Specificity"] = np.where(denominator > 0, frame["TN"] / denominator, 0.0)
    return frame


# ---------------------------------------------------------------- dataset
def fig_dataset_distribution() -> None:
    base = pd.read_excel(WORK_ROOT / "dataset" / "banglaplagpair1k_base.xlsx", engine="openpyxl")
    positive_types = ["Morphological", "Order", "Synonymy", "Lexical", "Mixed"]
    mismatch = base[base.pair_type.astype(str).str.lower().eq("mismatch")]
    labels = positive_types + ["Shared-ref.\nmismatch", "Same-source\nmismatch",
                               "Cross-source\nmismatch"]
    counts = [int((base.pair_type == c).sum()) for c in positive_types] + [
        int((mismatch.pair_id <= 600).sum()),
        int(((mismatch.pair_id > 600) & (mismatch.pair_id <= 800)).sum()),
        int((mismatch.pair_id > 800).sum()),
    ]
    colors = ["#2563eb"] * 5 + ["#dc2626", "#f97316", "#f59e0b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.25))
    bars = ax1.bar(labels, counts, color=colors, edgecolor="#1f2937", linewidth=0.5)
    ax1.bar_label(bars, fontsize=7, padding=2)
    ax1.set_ylabel("Pairs"); ax1.set_ylim(0, 230)
    ax1.set_title("Pair-type composition")
    ax1.tick_params(axis="x", rotation=28, labelsize=5.8)
    ax1.grid(axis="y", alpha=0.3)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")

    ch1 = base.text1.astype(str).map(len); ch2 = base.text2.astype(str).map(len)
    ax2.hist([ch1, ch2], bins=28, color=["#2563eb", "#f59e0b"],
             label=["text1", "text2"], edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Characters"); ax2.set_ylabel("Pairs")
    ax2.set_title("Passage-length distribution")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "dataset_distribution.png")


# ------------------------------------------------------- traditional track
def fig_raw_vs_polished_delta() -> None:
    d = pd.read_csv(RESULTS / "raw_vs_polished_delta.csv")
    d = d[d.Metric.isin(TRADITIONAL)]
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    b1 = ax.bar(x - w/2, d.Fixed_Raw_F1, w, label="Raw", color="#2563eb", edgecolor="#1f2937", linewidth=0.4)
    b2 = ax.bar(x + w/2, d.Fixed_Polished_F1, w, label="Polished", color="#f59e0b", edgecolor="#1f2937", linewidth=0.4)
    ax.bar_label(b1, fmt="%.3f", fontsize=6, padding=1)
    ax.bar_label(b2, fmt="%.3f", fontsize=6, padding=1)
    ax.set_xticks(x, d.Metric, rotation=15)
    ax.set_ylabel("F1 at $t=0.50$"); ax.set_ylim(0, 1.05)
    ax.set_title("Raw vs. polished traditional metrics")
    ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "raw_vs_polished_delta.png")


def fig_threshold_performance() -> None:
    sweep = pd.read_csv(RESULTS / "master_threshold_sweep.csv")
    raw = sweep[sweep.Stage == "Raw"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 4.9), sharex=True)
    for m in TRADITIONAL:
        g = raw[raw.Metric == m].sort_values("Threshold")
        ax1.plot(g.Threshold, g.F1, marker="o", ms=3.2, lw=1.45, label=m)
        ax2.plot(g.Threshold, g.Specificity, marker="o", ms=3.2, lw=1.45, label=m)
    ax1.set_xlabel("Threshold", fontsize=9); ax1.set_ylabel("F1", fontsize=9)
    ax1.set_title("F1 vs. threshold (raw)", fontsize=10)
    ax2.set_xlabel("Threshold", fontsize=9); ax2.set_ylabel("Specificity", fontsize=9)
    ax2.set_title("Specificity vs. threshold (raw)", fontsize=10)
    for a in (ax1, ax2):
        a.grid(alpha=0.3); a.set_ylim(0, 1.02); a.tick_params(labelsize=8)
    ax1.legend(ncol=2, fontsize=7.4)
    fig.tight_layout()
    save(fig, "threshold_performance_curve.png")


def fig_roc_curve() -> None:
    lab = pd.read_csv(BBERT / "normal" / "labeled_scores.csv")
    sent = pd.read_csv(SENT / "combined_sentence_embedding_evaluation" / "labeled_scores.csv")
    free = pd.read_csv(FREE / "combined_free_hf_model_evaluation" / "labeled_scores.csv")
    paid = pd.read_csv(PAID / "combined_llm_model_evaluation" / "labeled_scores.csv")
    y = lab.Label.values

    series = [("LCS (traditional)", lab.LCS.values, y, "#7c3aed"),
              ("BanglaBERT Sagor", lab.BanglaBERT_Cosine_SagorSarkar.values, y, "#0891b2"),
              ("LaBSE", sent.LaBSE_Score.values, sent.Label.values, "#16a34a"),
              ("Gemma 2 9B (shared prompt)", free.Gemma_2_9B_IT_Score.values, free.Label.values, "#2563eb"),
              ("Gemini 3.6 Flash (paid)", paid.Gemini_3_6_Flash_Score.values,
               paid.Label.values, "#dc2626")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for name, s, yy, c in series:
        order = np.argsort(-s)
        sy = yy[order]
        tpr = np.cumsum(sy) / max(sy.sum(), 1)
        fpr = np.cumsum(1 - sy) / max((1 - sy).sum(), 1)
        tpr = np.r_[0, tpr]; fpr = np.r_[0, fpr]
        auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
        ax1.plot(fpr, tpr, lw=1.3, color=c, label=f"{name} (AUC {auc:.3f})")
        prec, rec = [], []
        tp = fp = 0; pos = sy.sum()
        for v in sy:
            tp += v; fp += (1 - v)
            prec.append(tp / (tp + fp)); rec.append(tp / pos)
        ax2.plot(rec, prec, lw=1.3, color=c, label=name)
    ax1.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#9ca3af")
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate"); ax1.set_title("ROC")
    ax1.legend(fontsize=6, loc="lower right"); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-recall")
    ax2.grid(alpha=0.3); ax2.set_ylim(0.4, 1.02)
    fig.tight_layout()
    save(fig, "roc_curve.png")


# ------------------------------------------------------------- calibration
def fig_calibration_diagnostics() -> None:
    sweep = pd.read_csv(RESULTS / "master_threshold_sweep.csv")
    raw = sweep[sweep.Stage == "Raw"]
    sent_sweep = pd.read_csv(SENT / "combined_sentence_embedding_evaluation" / "threshold_sweep.csv")
    free_sweep = pd.read_csv(FREE / "combined_free_hf_model_evaluation" / "threshold_sweep.csv")
    paid_sweep = with_specificity(pd.read_csv(
        PAID / "combined_llm_model_evaluation" / "threshold_sweep.csv"))

    picks = [("LCS", raw, "#7c3aed"), ("BanglaBERT_Cosine_SagorSarkar", raw, "#0891b2"),
             ("LaBSE_Score", sent_sweep, "#16a34a"), ("BGE_M3_Score", sent_sweep, "#65a30d"),
             ("Gemma_2_9B_IT_Score", free_sweep, "#2563eb"),
             ("BanglaLLama_3_8B_Score", free_sweep, "#dc2626"),
             ("Gemini_3_6_Flash_Score", paid_sweep, "#9333ea")]
    nice = {"BanglaBERT_Cosine_SagorSarkar": "BanglaBERT Sagor", "LaBSE_Score": "LaBSE",
            "BGE_M3_Score": "BGE-M3", "Gemma_2_9B_IT_Score": "Gemma 2 9B",
            "BanglaLLama_3_8B_Score": "BanglaLLama 3 8B", "LCS": "LCS",
            "Gemini_3_6_Flash_Score": "Gemini 3.6 Flash"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9))
    for metric, src, c in picks:
        g = src[src.Metric == metric].sort_values("Threshold")
        if g.empty:
            continue
        ax1.plot(g.Threshold, g.F1, marker="o", ms=2.2, lw=1.2, color=c, label=nice.get(metric, metric))
        ax2.plot(g.Threshold, g.Specificity, marker="o", ms=2.2, lw=1.2, color=c, label=nice.get(metric, metric))
    ax1.axvline(0.5, ls="--", lw=0.8, color="#9ca3af")
    ax2.axvline(0.5, ls="--", lw=0.8, color="#9ca3af")
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("F1"); ax1.set_title("Threshold sensitivity: F1")
    ax2.set_xlabel("Threshold"); ax2.set_ylabel("Specificity"); ax2.set_title("Threshold sensitivity: specificity")
    for a in (ax1, ax2):
        a.grid(alpha=0.3); a.set_ylim(0, 1.03)
    ax1.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "calibration_diagnostics.png")


def fig_banglabert_score_distributions() -> None:
    """Compare class-conditional BanglaBERT cosine geometry by text condition."""
    panels = [
        ("Raw text - BUET", BBERT / "normal" / "labeled_scores.csv",
         "BanglaBERT_Cosine_BUETCSE", 0.90),
        ("Raw text - Sagor", BBERT / "normal" / "labeled_scores.csv",
         "BanglaBERT_Cosine_SagorSarkar", 0.85),
        ("Polished text - BUET", BBERT / "polished" / "labeled_scores.csv",
         "BanglaBERT_Cosine_BUETCSE", 0.90),
        ("Polished text - Sagor", BBERT / "polished" / "labeled_scores.csv",
         "BanglaBERT_Cosine_SagorSarkar", 0.90),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for ax, (title, path, metric, grid_t) in zip(axes.ravel(), panels):
        df = pd.read_csv(path)
        neg = pd.to_numeric(df.loc[df.Label.eq(0), metric], errors="coerce").dropna()
        pos = pd.to_numeric(df.loc[df.Label.eq(1), metric], errors="coerce").dropna()
        low, high = min(neg.min(), pos.min()), max(neg.max(), pos.max())
        pad = max((high - low) * 0.035, 0.003)
        bins = np.linspace(low - pad, min(1.001, high + pad), 30)
        ax.hist(neg, bins=bins, color="#f97316", alpha=0.68, label="Mismatch (y=0)")
        ax.hist(pos, bins=bins, color="#2563eb", alpha=0.58, label="Positive (y=1)")
        ax.axvline(0.50, color="#334155", linestyle="--", linewidth=0.9)
        ax.axvline(grid_t, color="#0f766e", linestyle=":", linewidth=1.2)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Cosine score")
        ax.set_ylabel("Pairs")
        ax.grid(axis="y", alpha=0.28)
    axes[0, 0].legend(frameon=True, fontsize=6.5)
    fig.suptitle("BanglaBERT score distributions on BanglaPlagPair-1K", y=0.995,
                 fontsize=10.2, fontweight="bold")
    fig.text(0.5, 0.008,
             "Dashed: fixed t=0.50   |   Dotted: highest-F1 point in the in-sample grid",
             ha="center", fontsize=7.0, color="#475569")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    save(fig, "banglabert_score_distributions.png")


def fig_representative_score_distributions() -> None:
    """Show model-specific score scales using current archived pair-level scores."""
    panels = [
        ("(A) BUET BanglaBERT", BBERT / "normal" / "labeled_scores.csv",
         "BanglaBERT_Cosine_BUETCSE", BBERT / "normal" / "best_threshold_metrics.csv"),
        ("(B) Multilingual E5-base",
         SENT / "combined_sentence_embedding_evaluation" / "labeled_scores.csv",
         "E5_Base_Score",
         SENT / "combined_sentence_embedding_evaluation" / "best_threshold_metrics.csv"),
        ("(C) LaBSE", SENT / "combined_sentence_embedding_evaluation" / "labeled_scores.csv",
         "LaBSE_Score",
         SENT / "combined_sentence_embedding_evaluation" / "best_threshold_metrics.csv"),
        ("(D) Gemma 2 9B IT (P0)",
         FREE / "combined_free_hf_model_evaluation" / "labeled_scores.csv",
         "Gemma_2_9B_IT_Score",
         FREE / "combined_free_hf_model_evaluation" / "best_threshold_metrics.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.1))
    for ax, (title, score_path, metric, result_path) in zip(axes.ravel(), panels):
        df = pd.read_csv(score_path)
        result = pd.read_csv(result_path)
        row = result[result.Metric.eq(metric)].iloc[0]
        grid_t = float(row.Threshold)
        neg = pd.to_numeric(df.loc[df.Label.eq(0), metric], errors="coerce").dropna().clip(0, 1)
        pos = pd.to_numeric(df.loc[df.Label.eq(1), metric], errors="coerce").dropna().clip(0, 1)
        low, high = min(neg.min(), pos.min()), max(neg.max(), pos.max())
        pad = max((high - low) * 0.035, 0.004)
        bins = np.linspace(max(0, low - pad), min(1.001, high + pad), 31)
        ax.hist(neg, bins=bins, color="#ef4444", alpha=0.67, label="Mismatch (y=0)")
        ax.hist(pos, bins=bins, color="#0284c7", alpha=0.60, label="Positive (y=1)")
        ax.axvline(0.50, color="#334155", linestyle="--", linewidth=0.9)
        ax.axvline(grid_t, color="#0f766e", linestyle=":", linewidth=1.25)
        ax.set_title(title, pad=4, fontweight="bold")
        ax.set_xlabel("Model score")
        ax.set_ylabel("Pairs")
        ax.grid(axis="y", alpha=0.28)
        ax.text(0.03, 0.96,
                f"grid t={grid_t:.2f} | F1={float(row.F1):.3f} | Spec.={float(row.Specificity):.3f}",
                transform=ax.transAxes, va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#94a3b8", alpha=0.92))
    axes[0, 0].legend(frameon=True, fontsize=6.5)
    fig.suptitle("Representative class-conditional score distributions", y=0.995,
                 fontsize=10.2, fontweight="bold")
    fig.text(0.5, 0.008,
             "Dashed: fixed t=0.50   |   Dotted: highest-F1 point in the in-sample grid",
             ha="center", fontsize=7.0, color="#475569")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    save(fig, "representative_score_distributions.png")


# -------------------------------------------------------- sentence + free
def fig_free_hf_score_separation() -> None:
    lab = pd.read_csv(FREE / "combined_free_hf_model_evaluation" / "labeled_scores.csv")
    cols = [c for c in lab.columns if c.endswith("_Score")]
    nice = [c.replace("_Score", "").replace("_", " ") for c in cols]
    pos = [lab.loc[lab.Label == 1, c].mean() for c in cols]
    neg = [lab.loc[lab.Label == 0, c].mean() for c in cols]
    x = np.arange(len(cols)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    b1 = ax.bar(x - w/2, pos, w, label="Positive pairs", color="#2563eb", edgecolor="#1f2937", linewidth=0.4)
    b2 = ax.bar(x + w/2, neg, w, label="Mismatch pairs", color="#dc2626", edgecolor="#1f2937", linewidth=0.4)
    ax.bar_label(b1, fmt="%.2f", fontsize=6, padding=1)
    ax.bar_label(b2, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(x, nice, rotation=28, ha="right")
    ax.set_ylabel("Mean score"); ax.set_ylim(0, 1.12)
    ax.set_title("Open-weight LLM score separation (shared prompt)")
    ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "free_hf_score_separation.png")


def fig_paid_llm_score_separation() -> None:
    lab = pd.read_csv(PAID / "combined_llm_model_evaluation" / "labeled_scores.csv")
    cols = [column for column in PAID_LABEL if column in lab.columns]
    pos = [lab.loc[lab.Label == 1, column].mean() for column in cols]
    neg = [lab.loc[lab.Label == 0, column].mean() for column in cols]
    x = np.arange(len(cols)); width = 0.38
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    positive_bars = ax.bar(
        x - width / 2, pos, width, label="Positive pairs", color="#2563eb",
        edgecolor="#1f2937", linewidth=0.4)
    negative_bars = ax.bar(
        x + width / 2, neg, width, label="Mismatch pairs", color="#dc2626",
        edgecolor="#1f2937", linewidth=0.4)
    ax.bar_label(positive_bars, fmt="%.2f", fontsize=6, padding=1)
    ax.bar_label(negative_bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(x, [PAID_LABEL[column] for column in cols], rotation=27, ha="right")
    ax.set_ylabel("Mean score"); ax.set_ylim(0, 1.18)
    ax.set_title("Paid-LLM score separation")
    ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "paid_llm_score_separation.png")


# ------------------------------------------------------------ prompt study
def fig_prompt_comparisons() -> None:
    fixed = pd.read_csv(PROMPT / "fixed_threshold_summary.csv")
    codes = [f"P{i}" for i in range(1, 9)]
    x = np.arange(len(codes)); w = 0.27

    for value, fname, title, ylab in [
        ("F1", "prompt_engineered_f1_comparison.png", "Prompt-engineering F1 at $t=0.50$", "F1"),
        ("Specificity", "prompt_engineered_specificity_comparison.png",
         "Prompt-engineering specificity at $t=0.50$", "Specificity"),
    ]:
        fig, ax = plt.subplots(figsize=(5.8, 4.15))
        for i, (model, color) in enumerate(MODEL_COLOR.items()):
            g = fixed[fixed.Model == model].set_index("Metric").reindex(codes)
            bars = ax.bar(x + (i - 1) * w, g[value].values, w, label=MODEL_SHORT[model],
                          color=color, edgecolor="#1f2937", linewidth=0.4)
            ax.bar_label(bars, fmt="%.2f", fontsize=5.7, padding=1)
        ax.set_xticks(x, [PROMPT_TICK[c] for c in codes], fontsize=6.6)
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel(ylab, fontsize=9); ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10)
        ax.legend(ncol=3, fontsize=7.0, loc="upper center",
                  bbox_to_anchor=(0.5, 1.02), frameon=True, framealpha=0.96,
                  borderaxespad=0.2, columnspacing=0.9, handlelength=1.5)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(rect=[0, 0.04, 1, 0.96])
        save(fig, fname)


def fig_selected_models_threshold_sensitivity() -> None:
    sweep = pd.read_csv(PROMPT / "threshold_sweep_summary.csv")
    fixed = pd.read_csv(PROMPT / "fixed_threshold_summary.csv")
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    markers = ["o", "s", "^"]
    for (model, color), marker in zip(MODEL_COLOR.items(), markers):
        best = fixed[fixed.Model == model].sort_values("F1", ascending=False).iloc[0]["Metric"]
        g = sweep[(sweep.Model == model) & (sweep.Metric == best)].sort_values("Threshold")
        ax.plot(g.Threshold, g.F1, color=color, marker=marker, markersize=3.2,
                markevery=2, linewidth=1.8, label=f"{MODEL_SHORT[model]} ({best})")
    ax.axvline(0.50, color="#64748b", linestyle="--", linewidth=1.0,
               label="Default threshold (0.50)")
    ax.set_title("Best prompt per selected model: threshold sensitivity")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.45, 1.015)
    ax.set_xticks(np.arange(0.10, 0.91, 0.10))
    ax.grid(alpha=0.35)
    ax.legend(loc="lower left", ncol=2, frameon=True, framealpha=0.96,
              edgecolor="#cbd5e1", fontsize=6.6)
    fig.tight_layout()
    save(fig, "selected_models_threshold_sensitivity.png")


# ---------------------------------------------------------- cross-family
def _cross_family_rows() -> pd.DataFrame:
    head = pd.read_csv(RESULTS / "headline_results.csv")
    raw = head[head.Stage == "Raw"].set_index("Metric")
    sent_f = pd.read_csv(SENT / "combined_sentence_embedding_evaluation" / "classification_metrics_threshold_0_50.csv").set_index("Metric")
    sent_b = pd.read_csv(SENT / "combined_sentence_embedding_evaluation" / "best_threshold_metrics.csv").set_index("Metric")
    free_f = pd.read_csv(FREE / "combined_free_hf_model_evaluation" / "classification_metrics_threshold_0_50.csv").set_index("Metric")
    free_b = pd.read_csv(FREE / "combined_free_hf_model_evaluation" / "best_threshold_metrics.csv").set_index("Metric")
    paid_f = with_specificity(pd.read_csv(
        PAID / "combined_llm_model_evaluation" / "classification_metrics_threshold_0_50.csv")).set_index("Metric")
    paid_b = with_specificity(pd.read_csv(
        PAID / "combined_llm_model_evaluation" / "best_threshold_metrics.csv")).set_index("Metric")
    pf = pd.read_csv(PROMPT / "fixed_threshold_summary.csv")
    pb = pd.read_csv(PROMPT / "best_threshold_summary.csv")

    best_trad = raw.loc[TRADITIONAL, "Fixed_F1"].idxmax()
    best_sent = sent_f.F1.idxmax()
    best_free = free_f.F1.idxmax()
    best_paid = paid_f.F1.idxmax()
    best_paid_grid = paid_b.F1.idxmax()
    prow = pf.sort_values("F1", ascending=False).iloc[0]
    pbest = pb.sort_values("F1", ascending=False).iloc[0]

    return pd.DataFrame([
        {"Family": "Traditional", "Config": f"{best_trad} (raw)",
         "F1": raw.loc[best_trad, "Fixed_F1"], "Specificity": raw.loc[best_trad, "Fixed_Specificity"],
         "BestF1": raw.loc[best_trad, "Calibrated_F1"]},
        {"Family": "BanglaBERT", "Config": "Sagor cosine",
         "F1": raw.loc["BanglaBERT_Cosine_SagorSarkar", "Fixed_F1"],
         "Specificity": raw.loc["BanglaBERT_Cosine_SagorSarkar", "Fixed_Specificity"],
         "BestF1": raw.loc["BanglaBERT_Cosine_SagorSarkar", "Calibrated_F1"]},
        {"Family": "Sentence emb.", "Config": SENT_LABEL.get(best_sent, best_sent),
         "F1": sent_f.loc[best_sent, "F1"], "Specificity": sent_f.loc[best_sent, "Specificity"],
         "BestF1": sent_b.loc[best_sent, "F1"]},
        {"Family": "Open LLM", "Config": best_free.replace("_Score", "").replace("_", " "),
         "F1": free_f.loc[best_free, "F1"], "Specificity": free_f.loc[best_free, "Specificity"],
         "BestF1": free_b.loc[best_free, "F1"]},
        {"Family": "Paid LLM", "Config": PAID_LABEL.get(best_paid, best_paid),
         "F1": paid_f.loc[best_paid, "F1"], "Specificity": paid_f.loc[best_paid, "Specificity"],
         "BestF1": paid_b.loc[best_paid_grid, "F1"]},
        {"Family": "Prompted LLM", "Config": f"{MODEL_SHORT[prow.Model]} {prow.Metric}",
         "F1": prow.F1, "Specificity": prow.Specificity, "BestF1": pbest.F1},
    ])


def fig_overall_f1_comparison() -> None:
    d = _cross_family_rows()
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    colors = ["#7c3aed", "#0891b2", "#16a34a", "#f59e0b", "#dc2626", "#2563eb"]
    bars = ax.bar(range(len(d)), d.F1, color=colors, edgecolor="#1f2937", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(range(len(d)), [f"{r.Family}\n{r.Config}" for r in d.itertuples()], fontsize=6.4)
    ax.set_ylabel("F1 at $t=0.50$"); ax.set_ylim(0, 1.1)
    ax.set_title("Best configuration per method family")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "overall_f1_comparison.png")


def fig_f1_vs_specificity() -> None:
    head = pd.read_csv(RESULTS / "headline_results.csv")
    raw = head[head.Stage == "Raw"]
    sent_f = pd.read_csv(SENT / "combined_sentence_embedding_evaluation" / "classification_metrics_threshold_0_50.csv")
    free_f = pd.read_csv(FREE / "combined_free_hf_model_evaluation" / "classification_metrics_threshold_0_50.csv")
    paid_f = with_specificity(pd.read_csv(
        PAID / "combined_llm_model_evaluation" / "classification_metrics_threshold_0_50.csv"))
    pf = pd.read_csv(PROMPT / "fixed_threshold_summary.csv")

    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    groups = [
        ("Traditional", raw[raw.Metric.isin(TRADITIONAL)].Fixed_F1, raw[raw.Metric.isin(TRADITIONAL)].Fixed_Specificity, "#7c3aed", "o"),
        ("BanglaBERT", raw[raw.Metric.str.startswith("BanglaBERT")].Fixed_F1, raw[raw.Metric.str.startswith("BanglaBERT")].Fixed_Specificity, "#0891b2", "s"),
        ("Sentence emb.", sent_f.F1, sent_f.Specificity, "#16a34a", "^"),
        ("Open LLM (shared)", free_f.F1, free_f.Specificity, "#f59e0b", "D"),
        ("Paid LLM", paid_f.F1, paid_f.Specificity, "#dc2626", "P"),
        ("Prompted LLM (P1-P8)", pf.F1, pf.Specificity, "#2563eb", "*"),
    ]
    for name, f1, spec, c, mk in groups:
        ax.scatter(spec, f1, s=42 if mk != "*" else 80, c=c, marker=mk,
                   edgecolors="#1f2937", linewidths=0.4, label=name, alpha=0.85, zorder=3)
    ax.set_xlabel("Specificity"); ax.set_ylabel("F1")
    ax.set_title("F1 vs. specificity at $t=0.50$, all evaluated configurations")
    ax.grid(alpha=0.3); ax.legend(fontsize=6.8, loc="lower right")
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    save(fig, "f1_vs_specificity_all_families.png")


# ------------------------------------------------------------- workflow
def fig_workflow_diagram() -> None:
    # Preserve the detailed vertical band style used in the previous report,
    # while keeping the diagram faithful to the current model coverage.
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ink = "#111827"

    def band(x, y, w, h, title, body, face, edge, title_size=7.6, body_size=5.8):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
            facecolor=face, edgecolor=edge, linewidth=1.25, zorder=2))
        if body:
            two_lines = "\n" in body
            title_y, body_y = ((0.75, 0.27) if two_lines else (0.63, 0.27))
            ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=ink, zorder=3)
            ax.text(x + w / 2, y + h * body_y, body, ha="center", va="center",
                    fontsize=body_size, color=ink, linespacing=1.35, zorder=3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=title_size, fontweight="bold", color=ink, zorder=3)

    def arrow(y_start, y_end, color="#475569"):
        ax.add_patch(FancyArrowPatch(
            (50, y_start), (50, y_end), arrowstyle="-|>", mutation_scale=11,
            linewidth=1.05, color=color, zorder=1))

    blue, orange, pink = "#dbeafe", "#fef0dc", "#fce4ec"
    green, grey, white = "#dcfce7", "#f1f5f9", "#ffffff"
    edge_blue, edge_orange = "#2563eb", "#ea8c1c"
    edge_pink, edge_green, edge_grey = "#d8628f", "#16a34a", "#64748b"

    ax.set_title("Specificity-Aware Pairwise Bangla Text-Reuse Evaluation",
                 fontsize=11.5, fontweight="bold", color=ink, pad=8)

    band(6, 91, 88, 6.8, "INPUT: 1,000 labelled Bangla text pairs",
         "500 controlled positive variants + 500 mismatch pairs",
         blue, edge_blue, 8.4, 6.3)
    arrow(91, 88.7)
    band(12, 81, 76, 7, "BENCHMARK CONSTRUCTION",
         "100 source passages -> 5 positive variants each  |  100 seed mismatches + 400 additional mismatches",
         orange, edge_orange, 8.2, 5.8)

    categories = [
        ("1. Morphological", "inflectional change"),
        ("2. Word order", "clause reorder"),
        ("3. Synonymy", "synonym substitution"),
        ("4. Lexical", "register shift"),
        ("5. Mixed", "combined changes"),
        ("6. Mismatch", "topical negatives"),
    ]
    category_width, category_gap = 14.2, 1.4
    category_x0 = (100 - (len(categories) * category_width +
                          (len(categories) - 1) * category_gap)) / 2
    for index, (title, body) in enumerate(categories):
        x = category_x0 + index * (category_width + category_gap)
        band(x, 70.5, category_width, 7, title, body, orange, edge_orange, 6.6, 5.0)
        ax.add_patch(FancyArrowPatch(
            (50, 80.7), (x + category_width / 2, 77.8), arrowstyle="-|>",
            mutation_scale=8, color=edge_orange, linewidth=0.8, zorder=1))

    arrow(70.2, 66.5, edge_pink)
    band(6, 56.6, 88, 8.5,
         "BANGLA-AWARE PREPROCESSING: RAW AND POLISHED TRACKS",
         "NFC | invisible-character removal | digit normalisation | punctuation removal\n"
         "lexical mapping | stopword removal | rule-based stemming | nine profiles",
         pink, edge_pink, 8.2, 5.8)
    arrow(56.6, 54.1, edge_blue)

    ax.add_patch(FancyBboxPatch(
        (4, 33.4), 92, 20, boxstyle="round,pad=0.35,rounding_size=1.2",
        facecolor=blue, edgecolor=edge_blue, linewidth=1.35, zorder=2))
    ax.text(50, 50.6, "FIVE SCORING FAMILIES - 28 SYSTEMS, ONE DECISION RULE",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=ink, zorder=3)
    families = [
        ("Lexical metrics (6)", "cosine, Jaccard, LCS,\nsequence, edit distance,\nword overlap"),
        ("Sentence embeddings (4)", "LaBSE, E5-base,\nBGE-M3, MiniLM\n(normalised cosine)"),
        ("Raw encoders (2)", "csebuetnlp/banglabert\nsagorsarker/bangla-bert-base\nmean-pooled L2 cosine"),
        ("Open-weight LLMs (8)", "Gemma 2, Llama 3,\nQwen 2.5, Aya Expanse\nshared P0 prompt"),
        ("Paid LLMs (8)", "DeepSeek V4, GPT 5.6,\nGemini 3.x, Qwen 3.7,\nClaude Sonnet 5, Grok 4.3"),
    ]
    family_width, family_gap = 16.7, 1.2
    family_x0 = (100 - (len(families) * family_width +
                        (len(families) - 1) * family_gap)) / 2
    for index, (title, body) in enumerate(families):
        x = family_x0 + index * (family_width + family_gap)
        ax.add_patch(FancyBboxPatch(
            (x, 35.1), family_width, 12.6,
            boxstyle="round,pad=0.28,rounding_size=0.9",
            facecolor=white, edgecolor="#93b4e8", linewidth=1.0, zorder=3))
        ax.text(x + family_width / 2, 45.1, title, ha="center", va="center",
                fontsize=5.9, fontweight="bold", color=ink, zorder=4)
        ax.text(x + family_width / 2, 39.6, body, ha="center", va="center",
                fontsize=4.7, color=ink, linespacing=1.30, zorder=4)

    arrow(33.4, 30.0, edge_orange)
    band(14, 22.5, 72, 7.2, "PROMPT-ENGINEERING EXTENSION",
         "top-3 open models x P1-P8 = 24 model-prompt configurations",
         orange, edge_orange, 8.2, 6.0)
    arrow(22.5, 19.0, edge_green)
    band(6, 10.2, 88, 8.8, "UNIFIED THRESHOLD-BASED EVALUATION",
         "fixed t = 0.50 | sweep t = 0.10...0.90 | TP / FP / TN / FN\n"
         "accuracy, precision, recall, F1, specificity, FPR, balanced accuracy",
         green, edge_green, 8.2, 5.8)
    arrow(10.2, 7.9, edge_grey)
    # Keep the final rounded border safely inside the canvas.  Placing it at
    # y=0.1 allowed the patch stroke to be clipped in the manuscript PDF.
    band(10, 1.2, 80, 6.6, "ANALYSIS AND RESEARCH OUTPUTS",
         "score files | threshold sweeps | confusion and subgroup diagnostics\n"
         "evidence for human-supervised screening research",
         grey, edge_grey, 8.0, 5.6)

    save(fig, "workflow_diagram.png")


# ----------------------------------------------------- preprocessing ablation
def fig_preprocessing_component_ablation() -> None:
    fixed = pd.read_csv(ABLATION / "fixed_threshold_summary.csv")
    profile_order = [
        ("raw", "Raw"), ("unicode_normalization", "Unicode"),
        ("lowercasing", "Lowercase"), ("digit_normalization", "Digits"),
        ("punctuation_removal", "Punctuation"), ("lexical_mapping", "Lexical map"),
        ("stopword_removal", "Stopwords"), ("stemming", "Stemming"),
        ("full_pipeline", "Full"),
    ]
    slugs = [slug for slug, _ in profile_order]
    labels = [label for _, label in profile_order]

    heat = (fixed.pivot(index="Profile", columns="Metric", values="Delta_F1_vs_Raw")
            .reindex(slugs)[TRADITIONAL])
    fig, ax = plt.subplots(figsize=(6.7, 3.45))
    im = ax.imshow(heat.values, cmap="RdYlGn", vmin=-0.03, vmax=0.03, aspect="auto")
    ax.set_xticks(np.arange(len(TRADITIONAL)), TRADITIONAL, rotation=25, ha="right", fontsize=6.7)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.7)
    ax.set_title("F1 change vs. raw text at $t=0.50$", fontsize=9.4)
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            ax.text(col, row, f"{heat.iat[row, col]:+.3f}",
                    ha="center", va="center", fontsize=5.2, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.022)
    cbar.set_label("Delta F1", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.3)
    fig.tight_layout()
    save(fig, "preprocessing_component_f1_delta_heatmap.png")

    rows = []
    for slug, label in profile_order:
        g = fixed[fixed.Profile.eq(slug)].sort_values(["F1", "Specificity"], ascending=False).iloc[0]
        rows.append({
            "Label": f"{label} ({g.Metric})",
            "F1": float(g.F1),
            "Specificity": float(g.Specificity),
        })
    best = pd.DataFrame(rows)
    y = np.arange(len(best))
    h = 0.34
    fig, ax = plt.subplots(figsize=(6.9, 4.15))
    f1_bars = ax.barh(y - h / 2, best.F1, h, color="#4e79a7", label="F1",
                      edgecolor="#1f2937", linewidth=0.35)
    spec_bars = ax.barh(y + h / 2, best.Specificity, h, color="#f28e2b",
                        label="Specificity", edgecolor="#1f2937", linewidth=0.35)
    ax.set_yticks(y, best.Label, fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.07)
    ax.set_xlabel("Score", fontsize=8.0)
    ax.set_title("Best fixed-threshold metric within each preprocessing setting", fontsize=9.2)
    ax.grid(axis="x", alpha=0.30)
    ax.bar_label(f1_bars, fmt="%.3f", fontsize=5.9, padding=2)
    ax.bar_label(spec_bars, fmt="%.3f", fontsize=5.9, padding=2)
    ax.legend(ncol=2, fontsize=7.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), frameon=True, framealpha=0.96,
              borderaxespad=0.2, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save(fig, "preprocessing_component_best_fixed_bar.png")


def main() -> None:
    style()
    print("Generating report figures ...")
    fig_dataset_distribution()
    fig_raw_vs_polished_delta()
    fig_threshold_performance()
    fig_roc_curve()
    fig_calibration_diagnostics()
    fig_banglabert_score_distributions()
    fig_representative_score_distributions()
    fig_free_hf_score_separation()
    fig_paid_llm_score_separation()
    fig_prompt_comparisons()
    fig_selected_models_threshold_sensitivity()
    fig_overall_f1_comparison()
    fig_f1_vs_specificity()
    fig_workflow_diagram()
    fig_preprocessing_component_ablation()
    print(f"\nAll figures in {FIG_DIR}")


if __name__ == "__main__":
    sys.exit(main())
