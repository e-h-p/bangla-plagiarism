# -*- coding: utf-8 -*-
"""Step 3 - apply the reference "full" polishing pipeline and rescore.

Preprocessing profile used for the reported polished condition:
    NFC unicode normalization + invisible-character removal
    -> lowercasing
    -> Bangla/Arabic-Indic digit normalization
    -> punctuation & symbol removal
    -> longest-match lexical normalization (confidence >= 0.85)
    -> curated stopword removal
    -> conservative rule-based Bangla stemming

The raw text is retained in ``text1_raw`` / ``text2_raw`` so the polished
artefact is fully auditable against the original. Identifiers, labels, source
links and row order are carried through untouched.
"""

from __future__ import annotations

import json

import pandas as pd

from pipeline_common import (
    METADATA_COLUMNS,
    METRICS,
    POLISHED_CSV,
    POLISHED_EVAL_DIR,
    POLISHED_SCORES_XLSX,
    POLISHED_SUMMARY_XLSX,
    POLISHED_XLSX,
    POLISH_CONFIG,
    POLISH_STEPS,
    VALIDATION_DIR,
    assert_structure_preserved,
    banner,
    build_preprocessor,
    ensure_dirs,
    evaluate_traditional,
    load_base_dataset,
    metric_distribution_summary,
    per_variant_summary,
    score_pairs,
    set_global_seed,
)


POLISHED_COLUMNS = [
    "pair_no",
    "variant_type",
    "text1_raw",
    "text2_raw",
    "text1",
    "text2",
    "text1_token_count",
    "text2_token_count",
    "preprocess_profile",
    "preprocess_steps",
] + METADATA_COLUMNS + ["Label"]


def polish(base: pd.DataFrame) -> pd.DataFrame:
    preprocessor = build_preprocessor()
    polished = base.copy()
    polished["text1_raw"] = polished["text1"]
    polished["text2_raw"] = polished["text2"]
    polished["text1"] = polished["text1_raw"].map(preprocessor.preprocess_text)
    polished["text2"] = polished["text2_raw"].map(preprocessor.preprocess_text)
    polished["preprocess_profile"] = "full"
    polished["preprocess_steps"] = POLISH_STEPS
    polished["text1_token_count"] = polished["text1"].map(lambda value: len(str(value).split()))
    polished["text2_token_count"] = polished["text2"].map(lambda value: len(str(value).split()))
    return polished[POLISHED_COLUMNS].copy()


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 3 - polished (preprocessed) dataset + similarity metrics")

    base = load_base_dataset()
    preprocessor = build_preprocessor()
    print(f"  Lexical map rules   : {len(preprocessor.lexical_normalizer.mapping)}")
    print(f"  Stopwords           : {len(preprocessor.stopwords)}")
    print(f"  Profile             : full ({POLISH_CONFIG})")

    polished = polish(base)

    # --- integrity guards -------------------------------------------------
    assert_structure_preserved(polished, base, context="polishing")
    if not polished["text1_raw"].astype(str).equals(base["text1"].astype(str)):
        raise RuntimeError("text1_raw does not match the original text1.")
    if not polished["text2_raw"].astype(str).equals(base["text2"].astype(str)):
        raise RuntimeError("text2_raw does not match the original text2.")

    empty1 = int((polished["text1"].astype(str).str.strip() == "").sum())
    empty2 = int((polished["text2"].astype(str).str.strip() == "").sum())
    if empty1 or empty2:
        raise RuntimeError(f"Polishing emptied {empty1} text1 and {empty2} text2 fields.")

    changed1 = int((polished["text1"].astype(str) != polished["text1_raw"].astype(str)).sum())
    changed2 = int((polished["text2"].astype(str) != polished["text2_raw"].astype(str)).sum())
    print(f"  Texts changed       : text1={changed1}/{len(polished)}, text2={changed2}/{len(polished)}")

    polished.to_excel(POLISHED_XLSX, index=False, engine="openpyxl")
    polished.to_csv(POLISHED_CSV, index=False, encoding="utf-8-sig")
    print(f"  Polished dataset    : {POLISHED_XLSX}")

    # --- rescore ----------------------------------------------------------
    print(f"  Recomputing {len(METRICS)} metrics on polished text ...")
    metric_frame = score_pairs(polished)
    scored = pd.concat([polished, metric_frame], axis=1)
    assert_structure_preserved(scored, base, context="polished scoring")

    for metric in METRICS:
        series = scored[metric]
        if series.isna().any():
            raise RuntimeError(f"Metric '{metric}' contains NaN values.")
        if float(series.min()) < 0.0 or float(series.max()) > 1.0:
            raise RuntimeError(f"Metric '{metric}' escaped the [0, 1] range.")

    scored.to_excel(POLISHED_SCORES_XLSX, index=False, engine="openpyxl")
    scored.to_csv(POLISHED_SCORES_XLSX.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"  Polished scores     : {POLISHED_SCORES_XLSX}")

    # --- summary workbook (same shape as the reference) -------------------
    summary = pd.DataFrame(
        [
            {
                "metric": metric,
                "min": round(float(scored[metric].min()), 6),
                "mean": round(float(scored[metric].mean()), 6),
                "max": round(float(scored[metric].max()), 6),
            }
            for metric in METRICS
        ]
    )
    token_stats = pd.DataFrame(
        [
            {
                "field": field,
                "min_tokens": int(scored[f"{field}_token_count"].min()),
                "mean_tokens": round(float(scored[f"{field}_token_count"].mean()), 3),
                "max_tokens": int(scored[f"{field}_token_count"].max()),
            }
            for field in ("text1", "text2")
        ]
    )
    with pd.ExcelWriter(POLISHED_SUMMARY_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="metric_summary", index=False)
        token_stats.to_excel(writer, sheet_name="token_summary", index=False)
        scored[["pair_no", "pair_type", "Label"] + METRICS].to_excel(
            writer, sheet_name="scores_compact", index=False
        )

    # --- evaluation -------------------------------------------------------
    labeled, class_df, sweep_df, best_df = evaluate_traditional(scored, POLISHED_EVAL_DIR)

    metric_distribution_summary(scored).to_csv(
        POLISHED_EVAL_DIR / "metric_distribution_summary.csv", index=False, encoding="utf-8-sig"
    )
    per_variant_summary(scored).to_csv(
        POLISHED_EVAL_DIR / "per_variant_metric_summary.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "stage": "polished",
        "rows": int(len(scored)),
        "profile": "full",
        "steps": POLISH_STEPS,
        "lexical_rules": len(preprocessor.lexical_normalizer.mapping),
        "stopwords": len(preprocessor.stopwords),
        "min_lexical_confidence": POLISH_CONFIG.min_lexical_confidence,
        "texts_changed": {"text1": changed1, "text2": changed2},
        "empty_after_polish": {"text1": empty1, "text2": empty2},
        "mean_tokens": {
            "text1": round(float(scored["text1_token_count"].mean()), 3),
            "text2": round(float(scored["text2_token_count"].mean()), 3),
        },
        "metric_ranges": {
            metric: {
                "min": round(float(scored[metric].min()), 6),
                "mean": round(float(scored[metric].mean()), 6),
                "max": round(float(scored[metric].max()), 6),
            }
            for metric in METRICS
        },
        "outputs": {
            "polished_xlsx": str(POLISHED_XLSX),
            "scores_xlsx": str(POLISHED_SCORES_XLSX),
            "evaluation_dir": str(POLISHED_EVAL_DIR),
        },
    }
    (VALIDATION_DIR / "step3_polished_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  --- Classification metrics @ t=0.50 (POLISHED) ---")
    print(class_df.to_string(index=False))
    print(f"\n  --- Best threshold per metric (POLISHED) ---")
    print(best_df.to_string(index=False))
    print(f"\n  Evaluation outputs  : {POLISHED_EVAL_DIR}")
    print("  [OK] Polished stage complete.")


if __name__ == "__main__":
    main()
