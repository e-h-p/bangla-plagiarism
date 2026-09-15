# -*- coding: utf-8 -*-
"""Shared plumbing for the BanglaPlagPair-1K (11 August 2026) experiment suite.

This module exposes the preprocessing, similarity, and evaluation components
vendored with the current project. The experiment suite is self-contained and
does not import code or resources from sibling legacy folders.

The original workbook ``BanglaPlagPair-1K_source_links_final.xlsx`` is only ever
opened read-only. Every artefact this suite produces is written to a separate
file under ``dataset/``, ``polish dataset/``, ``banglabert_evaluation/``,
``preprocessing_component_ablation/`` or ``validation/``.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════
# 1. Paths
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
WORK_ROOT = SCRIPT_DIR.parent
DATASET_DIR = WORK_ROOT / "dataset"

#: The pristine dataset. NEVER written to. Older runs kept the workbook at the
#: workspace root; the current project layout keeps it inside ``dataset``.
_ROOT_DATASET = WORK_ROOT / "BanglaPlagPair-1K_source_links_final.xlsx"
_DATASET_DIR_DATASET = DATASET_DIR / "BanglaPlagPair-1K_source_links_final.xlsx"
ORIGINAL_DATASET = _ROOT_DATASET if _ROOT_DATASET.exists() else _DATASET_DIR_DATASET

POLISH_DIR = WORK_ROOT / "polish dataset"
BANGLABERT_DIR = WORK_ROOT / "banglabert_evaluation"
ABLATION_DIR = WORK_ROOT / "preprocessing_component_ablation"
VALIDATION_DIR = WORK_ROOT / "validation"
RESULTS_DIR = WORK_ROOT / "results"

BASE_DATASET_XLSX = DATASET_DIR / "banglaplagpair1k_base.xlsx"
BASE_DATASET_CSV = DATASET_DIR / "banglaplagpair1k_base.csv"
RAW_SCORES_XLSX = DATASET_DIR / "banglaplagpair1k_similarity_scores.xlsx"
RAW_EVAL_DIR = DATASET_DIR / "raw_evaluation"

POLISHED_XLSX = POLISH_DIR / "banglaplagpair1k_polished.xlsx"
POLISHED_CSV = POLISH_DIR / "banglaplagpair1k_polished.csv"
POLISHED_SCORES_XLSX = POLISH_DIR / "banglaplagpair1k_similarity_scores_polished.xlsx"
POLISHED_SUMMARY_XLSX = POLISH_DIR / "banglaplagpair1k_polished_summary.xlsx"
POLISHED_EVAL_DIR = POLISH_DIR / "polished_evaluation"


# ═══════════════════════════════════════════════════════════════════
# 2. Local methodology imports
# ═══════════════════════════════════════════════════════════════════

from bangla_plagiarism_pipeline import (  # noqa: E402
    DEFAULT_LEXICAL_MAP,
    DEFAULT_STOPWORDS,
    METRICS,
    BanglaPreprocessor,
    PreprocessConfig,
    compute_metrics,
)
import evaluation_common as ec  # noqa: E402

FIXED_THRESHOLD = ec.FIXED_THRESHOLD          # 0.50
THRESHOLDS = list(ec.THRESHOLDS)              # 0.10 .. 0.90 step 0.05
classification_metrics = ec.classification_metrics
threshold_sweep = ec.threshold_sweep
best_thresholds = ec.best_thresholds
evaluate_scores = ec.evaluate_scores
write_evaluation_outputs = ec.write_evaluation_outputs


# ═══════════════════════════════════════════════════════════════════
# 3. Determinism
# ═══════════════════════════════════════════════════════════════════

SEED = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fix every RNG that could conceivably influence a result."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # torch is optional for the non-BERT stages
        pass


# ═══════════════════════════════════════════════════════════════════
# 4. Canonical dataset schema
# ═══════════════════════════════════════════════════════════════════

#: Columns exactly as they appear in the untouched source workbook.
ORIGINAL_COLUMNS = [
    "pair_id",
    "text1",
    "pair_type",
    "binary_label",
    "text2",
    "source_link_text1",
    "source_link_text2",
]

#: Canonical column order used by every downstream artefact. ``pair_no`` and
#: ``variant_type`` are aliases of ``pair_id`` / ``pair_type`` so that the
#: reference evaluation code (which keys on those names) runs unmodified.
BASE_COLUMNS = [
    "pair_no",
    "variant_type",
    "text1",
    "text2",
    "pair_id",
    "pair_type",
    "binary_label",
    "source_link_text1",
    "source_link_text2",
    "Label",
]

METADATA_COLUMNS = [
    "pair_id",
    "pair_type",
    "binary_label",
    "source_link_text1",
    "source_link_text2",
]

EXPECTED_ROWS = 1000


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_original_dataset() -> pd.DataFrame:
    """Read the pristine workbook. Read-only, no mutation, no reordering."""
    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(f"Missing original dataset: {ORIGINAL_DATASET}")

    frame = pd.read_excel(ORIGINAL_DATASET, sheet_name="Sheet1", engine="openpyxl")

    if list(frame.columns) != ORIGINAL_COLUMNS:
        raise RuntimeError(
            f"Unexpected columns in original dataset.\n"
            f"  expected: {ORIGINAL_COLUMNS}\n"
            f"  found:    {list(frame.columns)}"
        )
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows, found {len(frame)}")
    return frame


def build_base_frame(original: pd.DataFrame) -> pd.DataFrame:
    """Attach reference-compatible aliases and the binary Label column.

    ``text1`` / ``text2`` are carried through byte-for-byte. Row order is the
    original file order and is never sorted.
    """
    base = original.copy()

    base["pair_no"] = base["pair_id"]
    base["variant_type"] = base["pair_type"].astype(str)

    # Reference labelling rule: anything that is not "mismatch" is a positive.
    derived_label = base["variant_type"].str.strip().str.lower().ne("mismatch").astype(int)
    provided_label = base["binary_label"].astype(int)
    if not derived_label.equals(provided_label):
        disagreeing = base.loc[derived_label != provided_label, "pair_id"].tolist()
        raise RuntimeError(
            "binary_label disagrees with the reference labelling rule "
            f"(pair_type != 'mismatch') for pair_id: {disagreeing[:20]}"
        )
    base["Label"] = provided_label

    return base[BASE_COLUMNS].copy()


def load_base_dataset() -> pd.DataFrame:
    """Load the derived base dataset, falling back to the original workbook."""
    if BASE_DATASET_XLSX.exists():
        frame = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")
        for column in ("text1", "text2"):
            frame[column] = frame[column].fillna("").astype(str)
        return frame
    return build_base_frame(read_original_dataset())


# ═══════════════════════════════════════════════════════════════════
# 5. Preprocessing profile (identical to the reference "full" profile)
# ═══════════════════════════════════════════════════════════════════

POLISH_CONFIG = PreprocessConfig(
    unicode_form="NFC",
    lowercase=True,
    normalize_digits=True,
    remove_punctuation=True,
    apply_lexical_mapping=True,
    remove_stopwords=True,
    apply_stemming=True,
    min_lexical_confidence=0.85,
)

POLISH_STEPS = (
    "unicode_normalization + lowercasing + digit_normalization + punctuation_removal + "
    "lexical_mapping + stopword_removal + stemming"
)


def build_preprocessor() -> BanglaPreprocessor:
    """Reference ``full`` preprocessing profile with the reference resources."""
    return BanglaPreprocessor(
        POLISH_CONFIG,
        lexical_map_path=DEFAULT_LEXICAL_MAP,
        stopword_path=DEFAULT_STOPWORDS,
    )


# ═══════════════════════════════════════════════════════════════════
# 6. Scoring helper
# ═══════════════════════════════════════════════════════════════════

def score_pairs(frame: pd.DataFrame, *, text1: str = "text1", text2: str = "text2") -> pd.DataFrame:
    """Compute the six reference similarity metrics for every row, in order.

    Returns a frame with exactly ``len(frame)`` rows in the same order, indexed
    identically to ``frame``, containing only the six metric columns.
    """
    rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        a = "" if pd.isna(row[text1]) else str(row[text1])
        b = "" if pd.isna(row[text2]) else str(row[text2])
        scores = compute_metrics(a, b)
        rows.append({metric: round(scores[metric], 6) for metric in METRICS})
    return pd.DataFrame(rows, index=frame.index)


def assert_structure_preserved(frame: pd.DataFrame, reference: pd.DataFrame, *, context: str) -> None:
    """Guard: identifiers, labels, links and row order must be untouched."""
    if len(frame) != len(reference):
        raise RuntimeError(f"[{context}] row count changed: {len(frame)} vs {len(reference)}")

    for column in ("pair_no", "pair_id", "Label", "binary_label", "pair_type", "variant_type"):
        if column not in frame.columns or column not in reference.columns:
            continue
        left = frame[column].reset_index(drop=True)
        right = reference[column].reset_index(drop=True)
        if not left.astype(str).equals(right.astype(str)):
            raise RuntimeError(f"[{context}] column '{column}' was altered.")

    for column in ("source_link_text1", "source_link_text2"):
        if column not in frame.columns or column not in reference.columns:
            continue
        left = frame[column].reset_index(drop=True)
        right = reference[column].reset_index(drop=True)
        if not left.isna().equals(right.isna()):
            raise RuntimeError(f"[{context}] null pattern of '{column}' changed.")
        if not left.fillna("").astype(str).equals(right.fillna("").astype(str)):
            raise RuntimeError(f"[{context}] column '{column}' was altered.")


def evaluate_traditional(
    scored: pd.DataFrame,
    output_dir: Path,
    *,
    interactive_filename: str = "interactive_threshold_evaluation.xlsx",
    metrics: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the reference evaluation (t=0.50, sweep, best-threshold) and write outputs.

    Delegates to ``evaluation_common.evaluate_scores`` / ``write_evaluation_outputs``
    so the metric definitions, threshold grid and tie-breaking are identical to
    the previous experimental round.
    """
    metrics = list(metrics or METRICS)
    labeled, class_df, sweep_df, best_df = evaluate_scores(scored, metrics)
    write_evaluation_outputs(
        output_dir=output_dir,
        labeled_scores=labeled,
        class_df=class_df,
        sweep_df=sweep_df,
        best_df=best_df,
        metrics=metrics,
        interactive_filename=interactive_filename,
    )
    return labeled, class_df, sweep_df, best_df


def per_variant_summary(scored: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Mean/median similarity per ``pair_type`` for each metric."""
    metrics = list(metrics or METRICS)
    rows: list[dict[str, object]] = []
    for variant, group in scored.groupby("pair_type", sort=False):
        for metric in metrics:
            rows.append(
                {
                    "pair_type": variant,
                    "Label": int(group["Label"].iloc[0]),
                    "n": int(len(group)),
                    "Metric": metric,
                    "mean": round(float(group[metric].mean()), 6),
                    "median": round(float(group[metric].median()), 6),
                    "std": round(float(group[metric].std(ddof=1)), 6),
                    "min": round(float(group[metric].min()), 6),
                    "max": round(float(group[metric].max()), 6),
                }
            )
    return pd.DataFrame(rows)


def metric_distribution_summary(scored: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Overall / positive / negative score distribution and separation per metric."""
    metrics = list(metrics or METRICS)
    positives = scored[scored["Label"] == 1]
    negatives = scored[scored["Label"] == 0]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        pos_mean = float(positives[metric].mean())
        neg_mean = float(negatives[metric].mean())
        pos_std = float(positives[metric].std(ddof=1))
        neg_std = float(negatives[metric].std(ddof=1))
        pooled = ((pos_std**2 + neg_std**2) / 2) ** 0.5
        rows.append(
            {
                "Metric": metric,
                "Overall_Min": round(float(scored[metric].min()), 6),
                "Overall_Mean": round(float(scored[metric].mean()), 6),
                "Overall_Max": round(float(scored[metric].max()), 6),
                "Positive_Mean": round(pos_mean, 6),
                "Positive_Std": round(pos_std, 6),
                "Negative_Mean": round(neg_mean, 6),
                "Negative_Std": round(neg_std, 6),
                "Mean_Separation": round(pos_mean - neg_mean, 6),
                "Cohens_d": round((pos_mean - neg_mean) / pooled, 6) if pooled else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ensure_dirs() -> None:
    for directory in (
        DATASET_DIR,
        POLISH_DIR,
        BANGLABERT_DIR,
        ABLATION_DIR,
        VALIDATION_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def banner(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)
