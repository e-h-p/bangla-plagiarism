# -*- coding: utf-8 -*-
"""Step 2 - score the RAW (unmodified) dataset with the six reference metrics.

Metrics (implemented in the local ``bangla_plagiarism_pipeline.py``):
    Cosine       TF-IDF cosine over Unicode-safe Bangla tokens
    Jaccard      token-set Jaccard
    LCS          character LCS length / max(len)
    Sequence     difflib SequenceMatcher ratio
    EditDist     1 - (Levenshtein distance / max(len))
    WordOverlap  |A n B| / min(|A|, |B|)

Text is used exactly as it appears in the source workbook - no normalization is
applied at this stage.
"""

from __future__ import annotations

import json

import pandas as pd

from pipeline_common import (
    BASE_COLUMNS,
    METRICS,
    RAW_EVAL_DIR,
    RAW_SCORES_XLSX,
    DATASET_DIR,
    VALIDATION_DIR,
    assert_structure_preserved,
    banner,
    ensure_dirs,
    evaluate_traditional,
    load_base_dataset,
    metric_distribution_summary,
    per_variant_summary,
    score_pairs,
    set_global_seed,
)


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 2 - traditional similarity metrics on the RAW dataset")

    base = load_base_dataset()
    print(f"  Loaded {len(base)} pairs from the base dataset.")

    print(f"  Computing {len(METRICS)} metrics per pair ...")
    metric_frame = score_pairs(base)
    scored = pd.concat([base, metric_frame], axis=1)

    # --- integrity guards -------------------------------------------------
    assert_structure_preserved(scored, base, context="raw scoring")
    if list(scored.columns) != BASE_COLUMNS + METRICS:
        raise RuntimeError(f"Unexpected scored column order: {list(scored.columns)}")
    for metric in METRICS:
        series = scored[metric]
        if series.isna().any():
            raise RuntimeError(f"Metric '{metric}' contains NaN values.")
        if float(series.min()) < 0.0 or float(series.max()) > 1.0:
            raise RuntimeError(f"Metric '{metric}' escaped the [0, 1] range.")

    scored.to_excel(RAW_SCORES_XLSX, index=False, engine="openpyxl")
    scored.to_csv(DATASET_DIR / "banglaplagpair1k_similarity_scores.csv", index=False, encoding="utf-8-sig")
    print(f"  Similarity workbook : {RAW_SCORES_XLSX}")

    # --- evaluation -------------------------------------------------------
    labeled, class_df, sweep_df, best_df = evaluate_traditional(scored, RAW_EVAL_DIR)

    dist = metric_distribution_summary(scored)
    variant = per_variant_summary(scored)
    dist.to_csv(RAW_EVAL_DIR / "metric_distribution_summary.csv", index=False, encoding="utf-8-sig")
    variant.to_csv(RAW_EVAL_DIR / "per_variant_metric_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "stage": "raw",
        "rows": int(len(scored)),
        "metrics": METRICS,
        "positives": int(scored["Label"].sum()),
        "negatives": int(len(scored) - scored["Label"].sum()),
        "metric_ranges": {
            metric: {
                "min": round(float(scored[metric].min()), 6),
                "mean": round(float(scored[metric].mean()), 6),
                "max": round(float(scored[metric].max()), 6),
            }
            for metric in METRICS
        },
        "outputs": {
            "scores_xlsx": str(RAW_SCORES_XLSX),
            "evaluation_dir": str(RAW_EVAL_DIR),
        },
    }
    (VALIDATION_DIR / "step2_raw_scoring_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  --- Classification metrics @ t=0.50 (RAW) ---")
    print(class_df.to_string(index=False))
    print(f"\n  --- Best threshold per metric (RAW) ---")
    print(best_df.to_string(index=False))
    print(f"\n  Evaluation outputs  : {RAW_EVAL_DIR}")
    print("  [OK] Raw stage complete.")


if __name__ == "__main__":
    main()
