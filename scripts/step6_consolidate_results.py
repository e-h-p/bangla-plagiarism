# -*- coding: utf-8 -*-
"""Step 6 - consolidate every result table and run the final integrity audit.

Produces ``results/`` with cross-stage master tables and
``validation/final_integrity_report.json`` with the end-to-end audit:

  * the original workbook is byte-identical to its pre-experiment SHA-256
  * every artefact carries the same 1000 pair_ids, in the same order
  * labels, pair types and source links are identical everywhere
  * traditional-metric numbers agree between the standalone evaluation and the
    copy carried into the BanglaBERT evaluation
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd

from pipeline_common import (
    BANGLABERT_DIR,
    BASE_DATASET_XLSX,
    ABLATION_DIR,
    FIXED_THRESHOLD,
    METRICS,
    ORIGINAL_DATASET,
    POLISHED_EVAL_DIR,
    POLISHED_SCORES_XLSX,
    RAW_EVAL_DIR,
    RAW_SCORES_XLSX,
    RESULTS_DIR,
    SEED,
    THRESHOLDS,
    VALIDATION_DIR,
    WORK_ROOT,
    banner,
    ensure_dirs,
    set_global_seed,
    sha256_of,
)


BERT_METRICS = ["BanglaBERT_Cosine_BUETCSE", "BanglaBERT_Cosine_SagorSarkar"]
ALL_METRICS = BERT_METRICS + METRICS

METRIC_FAMILY = {
    **{m: "Traditional" for m in METRICS},
    "BanglaBERT_Cosine_BUETCSE": "BanglaBERT (csebuetnlp/banglabert)",
    "BanglaBERT_Cosine_SagorSarkar": "BanglaBERT (sagorsarker/bangla-bert-base)",
}


def tag(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "Stage", stage)
    out.insert(1, "Family", out["Metric"].map(METRIC_FAMILY))
    return out


def load_stage_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Master fixed / best / sweep tables across both dataset states."""
    fixed_parts, best_parts, sweep_parts = [], [], []

    for stage, directory in (("Raw", BANGLABERT_DIR / "normal"), ("Polished", BANGLABERT_DIR / "polished")):
        fixed_parts.append(tag(pd.read_csv(directory / "classification_metrics_threshold_0_50.csv"), stage))
        best_parts.append(tag(pd.read_csv(directory / "best_threshold_metrics.csv"), stage))
        sweep_parts.append(tag(pd.read_csv(directory / "threshold_sweep.csv"), stage))

    order = {metric: i for i, metric in enumerate(ALL_METRICS)}
    fixed = pd.concat(fixed_parts, ignore_index=True)
    best = pd.concat(best_parts, ignore_index=True)
    sweep = pd.concat(sweep_parts, ignore_index=True)

    for frame in (fixed, best):
        frame["_order"] = frame["Metric"].map(order)
        frame.sort_values(["Stage", "_order"], ascending=[False, True], inplace=True)
        frame.drop(columns="_order", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    return fixed, best, sweep


def raw_vs_polished(fixed: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Delta table: polished minus raw, at the fixed and at the calibrated threshold."""
    cols = ["Accuracy", "Precision", "Recall", "Specificity", "FPR", "BalancedAccuracy", "F1"]
    rows: list[dict[str, object]] = []

    for metric in ALL_METRICS:
        fr = fixed[(fixed["Stage"] == "Raw") & (fixed["Metric"] == metric)].iloc[0]
        fp = fixed[(fixed["Stage"] == "Polished") & (fixed["Metric"] == metric)].iloc[0]
        br = best[(best["Stage"] == "Raw") & (best["Metric"] == metric)].iloc[0]
        bp = best[(best["Stage"] == "Polished") & (best["Metric"] == metric)].iloc[0]

        record: dict[str, object] = {"Metric": metric, "Family": METRIC_FAMILY[metric]}
        for col in cols:
            record[f"Fixed_Raw_{col}"] = fr[col]
            record[f"Fixed_Polished_{col}"] = fp[col]
            record[f"Fixed_Delta_{col}"] = round(float(fp[col]) - float(fr[col]), 6)
        record["Best_Raw_Threshold"] = br["Threshold"]
        record["Best_Polished_Threshold"] = bp["Threshold"]
        for col in cols:
            record[f"Best_Raw_{col}"] = br[col]
            record[f"Best_Polished_{col}"] = bp[col]
            record[f"Best_Delta_{col}"] = round(float(bp[col]) - float(br[col]), 6)
        rows.append(record)

    return pd.DataFrame(rows)


def headline_table(fixed: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Compact publication-style table: one row per (stage, metric)."""
    rows: list[dict[str, object]] = []
    for stage in ("Raw", "Polished"):
        for metric in ALL_METRICS:
            f = fixed[(fixed["Stage"] == stage) & (fixed["Metric"] == metric)].iloc[0]
            b = best[(best["Stage"] == stage) & (best["Metric"] == metric)].iloc[0]
            rows.append(
                {
                    "Stage": stage,
                    "Family": METRIC_FAMILY[metric],
                    "Metric": metric,
                    "Fixed_Threshold": FIXED_THRESHOLD,
                    "Fixed_Accuracy": f["Accuracy"],
                    "Fixed_Precision": f["Precision"],
                    "Fixed_Recall": f["Recall"],
                    "Fixed_Specificity": f["Specificity"],
                    "Fixed_F1": f["F1"],
                    "Calibrated_Threshold": b["Threshold"],
                    "Calibrated_Accuracy": b["Accuracy"],
                    "Calibrated_Precision": b["Precision"],
                    "Calibrated_Recall": b["Recall"],
                    "Calibrated_Specificity": b["Specificity"],
                    "Calibrated_F1": b["F1"],
                }
            )
    return pd.DataFrame(rows)


def score_distributions() -> pd.DataFrame:
    """Positive/negative separation for every metric in both dataset states."""
    rows: list[dict[str, object]] = []
    for stage, directory in (("Raw", BANGLABERT_DIR / "normal"), ("Polished", BANGLABERT_DIR / "polished")):
        scores = pd.read_csv(directory / "labeled_scores.csv")
        pos = scores[scores["Label"] == 1]
        neg = scores[scores["Label"] == 0]
        for metric in ALL_METRICS:
            pos_mean, neg_mean = float(pos[metric].mean()), float(neg[metric].mean())
            pos_std, neg_std = float(pos[metric].std(ddof=1)), float(neg[metric].std(ddof=1))
            pooled = ((pos_std**2 + neg_std**2) / 2) ** 0.5
            rows.append(
                {
                    "Stage": stage,
                    "Family": METRIC_FAMILY[metric],
                    "Metric": metric,
                    "Min": round(float(scores[metric].min()), 6),
                    "Mean": round(float(scores[metric].mean()), 6),
                    "Max": round(float(scores[metric].max()), 6),
                    "Std": round(float(scores[metric].std(ddof=1)), 6),
                    "Positive_Mean": round(pos_mean, 6),
                    "Negative_Mean": round(neg_mean, 6),
                    "Mean_Separation": round(pos_mean - neg_mean, 6),
                    "Cohens_d": round((pos_mean - neg_mean) / pooled, 6) if pooled else 0.0,
                }
            )
    return pd.DataFrame(rows)


def per_variant_recall() -> pd.DataFrame:
    """Recall per pair_type at t=0.50 - shows which paraphrase families are hard."""
    rows: list[dict[str, object]] = []
    for stage, directory in (("Raw", BANGLABERT_DIR / "normal"), ("Polished", BANGLABERT_DIR / "polished")):
        scores = pd.read_csv(directory / "labeled_scores.csv")
        for variant, group in scores.groupby("pair_type", sort=False):
            is_positive = int(group["Label"].iloc[0]) == 1
            for metric in ALL_METRICS:
                flagged = int((group[metric] >= FIXED_THRESHOLD).sum())
                rows.append(
                    {
                        "Stage": stage,
                        "pair_type": variant,
                        "Label": int(group["Label"].iloc[0]),
                        "n": int(len(group)),
                        "Family": METRIC_FAMILY[metric],
                        "Metric": metric,
                        "Mean_Score": round(float(group[metric].mean()), 6),
                        "Flagged_At_0.50": flagged,
                        # positives -> recall; mismatch rows -> false-positive rate
                        "Rate": round(flagged / len(group), 6),
                        "Rate_Meaning": "Recall" if is_positive else "FalsePositiveRate",
                    }
                )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Integrity audit
# ═══════════════════════════════════════════════════════════════════

def audit() -> dict[str, object]:
    base = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")
    expected_ids = base["pair_id"].tolist()
    expected_labels = base["Label"].astype(int).tolist()
    expected_types = base["pair_type"].astype(str).tolist()
    expected_link1 = base["source_link_text1"].fillna("").astype(str).tolist()
    expected_link2 = base["source_link_text2"].fillna("").astype(str).tolist()
    expected_text1 = base["text1"].astype(str).tolist()
    expected_text2 = base["text2"].astype(str).tolist()

    artefacts = {
        "raw_scores": RAW_SCORES_XLSX,
        "polished_scores": POLISHED_SCORES_XLSX,
        "raw_labeled_scores": RAW_EVAL_DIR / "labeled_scores.csv",
        "polished_labeled_scores": POLISHED_EVAL_DIR / "labeled_scores.csv",
        "banglabert_normal": BANGLABERT_DIR / "normal" / "labeled_scores.csv",
        "banglabert_polished": BANGLABERT_DIR / "polished" / "labeled_scores.csv",
    }

    results: dict[str, object] = {}
    for name, path in artefacts.items():
        frame = pd.read_excel(path, engine="openpyxl") if path.suffix == ".xlsx" else pd.read_csv(path)
        checks = {
            "path": str(path),
            "rows": int(len(frame)),
            "row_count_ok": len(frame) == len(base),
            "pair_id_order_ok": frame["pair_id"].tolist() == expected_ids,
            "pair_no_order_ok": frame["pair_no"].tolist() == expected_ids,
            "label_ok": frame["Label"].astype(int).tolist() == expected_labels,
            "binary_label_ok": frame["binary_label"].astype(int).tolist() == expected_labels,
            "pair_type_ok": frame["pair_type"].astype(str).tolist() == expected_types,
            "source_link_text1_ok": frame["source_link_text1"].fillna("").astype(str).tolist() == expected_link1,
            "source_link_text2_ok": frame["source_link_text2"].fillna("").astype(str).tolist() == expected_link2,
        }
        # Raw-text preservation: raw artefacts keep text1/text2; polished keep text*_raw.
        if "text1_raw" in frame.columns:
            checks["original_text_preserved_ok"] = (
                frame["text1_raw"].astype(str).tolist() == expected_text1
                and frame["text2_raw"].astype(str).tolist() == expected_text2
            )
            checks["text_source"] = "text1_raw / text2_raw"
        else:
            checks["original_text_preserved_ok"] = (
                frame["text1"].astype(str).tolist() == expected_text1
                and frame["text2"].astype(str).tolist() == expected_text2
            )
            checks["text_source"] = "text1 / text2"
        results[name] = checks

    # Cross-check: traditional metrics must be identical in the standalone
    # evaluation and in the copy carried into the BanglaBERT evaluation.
    consistency: dict[str, object] = {}
    for stage, standalone, bert in (
        ("Raw", RAW_EVAL_DIR, BANGLABERT_DIR / "normal"),
        ("Polished", POLISHED_EVAL_DIR, BANGLABERT_DIR / "polished"),
    ):
        a = pd.read_csv(standalone / "classification_metrics_threshold_0_50.csv").set_index("Metric")
        b = pd.read_csv(bert / "classification_metrics_threshold_0_50.csv").set_index("Metric")
        cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "TP", "FP", "TN", "FN"]
        diffs = {}
        for metric in METRICS:
            delta = {c: float(a.loc[metric, c]) - float(b.loc[metric, c]) for c in cols}
            worst = max(abs(v) for v in delta.values())
            diffs[metric] = round(worst, 9)
        consistency[stage] = {
            "max_abs_difference_per_metric": diffs,
            "consistent": all(v < 1e-9 for v in diffs.values()),
        }

    return {"artefacts": results, "traditional_metric_consistency": consistency}


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 6 - consolidate results and audit integrity")

    fixed, best, sweep = load_stage_tables()
    delta = raw_vs_polished(fixed, best)
    headline = headline_table(fixed, best)
    distributions = score_distributions()
    variants = per_variant_recall()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(RESULTS_DIR / "master_fixed_threshold_results.csv", index=False, encoding="utf-8-sig")
    best.to_csv(RESULTS_DIR / "master_best_threshold_results.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(RESULTS_DIR / "master_threshold_sweep.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(RESULTS_DIR / "raw_vs_polished_delta.csv", index=False, encoding="utf-8-sig")
    headline.to_csv(RESULTS_DIR / "headline_results.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(RESULTS_DIR / "score_distribution_summary.csv", index=False, encoding="utf-8-sig")
    variants.to_csv(RESULTS_DIR / "per_variant_detection_rates.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "banglaplagpair1k_results_summary.xlsx", engine="openpyxl") as writer:
        headline.to_excel(writer, sheet_name="headline", index=False)
        fixed.to_excel(writer, sheet_name="fixed_threshold_0_50", index=False)
        best.to_excel(writer, sheet_name="best_threshold", index=False)
        delta.to_excel(writer, sheet_name="raw_vs_polished_delta", index=False)
        distributions.to_excel(writer, sheet_name="score_distributions", index=False)
        variants.to_excel(writer, sheet_name="per_variant_rates", index=False)
        sweep.to_excel(writer, sheet_name="threshold_sweep", index=False)

    integrity = audit()
    original_hash = sha256_of(ORIGINAL_DATASET)
    step1 = json.loads((VALIDATION_DIR / "step1_base_dataset_validation.json").read_text(encoding="utf-8"))
    integrity["original_dataset"] = {
        "path": str(ORIGINAL_DATASET),
        "sha256_at_step1": step1["original_sha256"],
        "sha256_now": original_hash,
        "unchanged": original_hash == step1["original_sha256"],
    }

    failed = []
    for name, checks in integrity["artefacts"].items():
        failed += [f"{name}.{k}" for k, v in checks.items() if k.endswith("_ok") and v is False]
    for stage, info in integrity["traditional_metric_consistency"].items():
        if not info["consistent"]:
            failed.append(f"traditional_metric_consistency.{stage}")
    if not integrity["original_dataset"]["unchanged"]:
        failed.append("original_dataset.unchanged")
    integrity["failed_checks"] = failed
    integrity["all_passed"] = not failed

    import numpy as np
    import sklearn
    import torch
    import transformers

    integrity["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "seed": SEED,
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_sweep": THRESHOLDS,
    }

    (VALIDATION_DIR / "final_integrity_report.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # -- manifest of every produced artefact --
    manifest = []
    for path in sorted(WORK_ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            manifest.append(
                {
                    "path": str(path.relative_to(WORK_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_of(path) if path.stat().st_size < 50_000_000 else None,
                }
            )
    (VALIDATION_DIR / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n  --- HEADLINE RESULTS ---")
    print(headline.to_string(index=False))
    print("\n  --- POLISHED minus RAW (F1 @ t=0.50) ---")
    print(delta[["Metric", "Fixed_Raw_F1", "Fixed_Polished_F1", "Fixed_Delta_F1"]].to_string(index=False))
    print(f"\n  Original dataset unchanged: {integrity['original_dataset']['unchanged']}")
    print(f"  Integrity checks passed   : {integrity['all_passed']}")
    if failed:
        print(f"  FAILED: {failed}")
    print(f"\n  Result tables : {RESULTS_DIR}")
    print(f"  Audit report  : {VALIDATION_DIR / 'final_integrity_report.json'}")


if __name__ == "__main__":
    main()
