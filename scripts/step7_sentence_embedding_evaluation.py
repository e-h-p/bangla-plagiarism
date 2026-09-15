# -*- coding: utf-8 -*-
"""Step 7 - sentence-embedding model evaluation on BanglaPlagPair-1K.

Evaluates the archived sentence-embedding score
workbook supplied for the updated dataset. The reference module is imported and
its path constants are redirected at run time, so the evaluation logic, the
0-1 clipping, the audit/distribution/category tables and the output layout are
identical to the previous round.

Models evaluated (scores supplied in the input workbook):
    LaBSE_Score, E5_Base_Score, BGE_M3_Score, MiniLM_Multilingual_Score

The only adaptation is ``id_column="pair_id"``: the new workbook keys rows on
``pair_id`` where the old one used ``pair_no``. The values are identical, and
``normalize_model_scores`` maps it back onto ``pair_no`` before merging.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from pipeline_common import (
    BASE_DATASET_XLSX,
    VALIDATION_DIR,
    WORK_ROOT,
    banner,
    ensure_dirs,
    set_global_seed,
)

import evaluate_sentence_embedding_models as sem


OUTPUT_ROOT = WORK_ROOT / "Sentence embedding models evualtion"
# Rescored after the text2 corrections to pair_id 12, 354 and 668.
# (previous input: "combined_1000_testcases_ scores.xlsx", scored on the old text2)
INPUT_FILE = OUTPUT_ROOT / "combined_1000_testcases_ scores3.xlsx"


def redirect_reference_paths() -> None:
    """Point the reference module at this working directory and input workbook."""
    sem.ROOT = WORK_ROOT
    sem.BASE_FILE = BASE_DATASET_XLSX
    sem.INPUT_FILE = INPUT_FILE
    sem.OUTPUT_ROOT = OUTPUT_ROOT
    # Rebuild the configs so they point at the new workbook and key on pair_id.
    sem.CONFIGS = [
        replace(config, input_file=INPUT_FILE, id_column="pair_id")
        for config in sem.CONFIGS
    ]


def preflight() -> dict[str, object]:
    """Confirm the supplied workbook lines up with the untouched base dataset."""
    scores = pd.read_excel(INPUT_FILE, engine="openpyxl")
    base = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")

    metrics = [config.metric_column for config in sem.CONFIGS]
    missing = [m for m in metrics if m not in scores.columns]
    if missing:
        raise RuntimeError(f"Input workbook is missing score columns: {missing}")

    checks = {
        "input_file": str(INPUT_FILE),
        "rows": int(len(scores)),
        "row_count_ok": len(scores) == len(base),
        "pair_id_unique": bool(scores["pair_id"].is_unique),
        "pair_id_order_ok": scores["pair_id"].tolist() == base["pair_id"].tolist(),
        "text1_identical": bool(scores["text1"].astype(str).equals(base["text1"].astype(str))),
        "text2_identical": bool(scores["text2"].astype(str).equals(base["text2"].astype(str))),
        "score_columns": metrics,
        "raw_score_ranges": {
            m: {
                "min": round(float(scores[m].min()), 6),
                "max": round(float(scores[m].max()), 6),
                "missing": int(scores[m].isna().sum()),
                "below_0": int((scores[m] < 0).sum()),
                "above_1": int((scores[m] > 1).sum()),
            }
            for m in metrics
        },
    }

    failed = [k for k in ("row_count_ok", "pair_id_unique", "pair_id_order_ok",
                          "text1_identical", "text2_identical") if checks[k] is False]
    if failed:
        raise RuntimeError(f"Input workbook does not match the base dataset: {failed}")
    return checks


def postflight() -> dict[str, object]:
    """Confirm every produced evaluation covers the same 1000 rows and labels."""
    base = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")
    expected_ids = base["pair_id"].tolist()
    expected_labels = base["Label"].astype(int).tolist()

    per_model: dict[str, object] = {}
    directories = [f"{config.slug}_evaluation" for config in sem.CONFIGS]
    directories.append("combined_sentence_embedding_evaluation")

    for name in directories:
        labeled = pd.read_csv(OUTPUT_ROOT / name / "labeled_scores.csv")
        if labeled["pair_no"].tolist() != expected_ids:
            raise RuntimeError(f"[{name}] pair_no order changed.")
        if labeled["Label"].astype(int).tolist() != expected_labels:
            raise RuntimeError(f"[{name}] labels changed.")
        per_model[name] = {"rows": int(len(labeled)), "positives": int(labeled["Label"].sum())}

    return per_model


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 7 - sentence-embedding model evaluation")

    redirect_reference_paths()
    print(f"  Base dataset : {sem.BASE_FILE}")
    print(f"  Input scores : {sem.INPUT_FILE}")
    print(f"  Output root  : {sem.OUTPUT_ROOT}")
    print(f"  Models       : {[c.model_name for c in sem.CONFIGS]}\n")

    checks = preflight()
    print("  Preflight: input workbook matches the base dataset (ids, order, texts).")
    for metric, info in checks["raw_score_ranges"].items():
        note = ""
        if info["below_0"] or info["above_1"]:
            note = f"  <- {info['below_0']} below 0, {info['above_1']} above 1 (will be clipped)"
        print(f"    {metric:28s} [{info['min']:.6f}, {info['max']:.6f}]{note}")
    print()

    sem.main()

    checks["per_output"] = postflight()
    (VALIDATION_DIR / "step7_sentence_embedding_validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  --- Combined evaluation @ t=0.50 ---")
    print(pd.read_csv(
        OUTPUT_ROOT / "combined_sentence_embedding_evaluation" / "classification_metrics_threshold_0_50.csv"
    ).to_string(index=False))
    print(f"\n  --- Best threshold per model ---")
    print(pd.read_csv(
        OUTPUT_ROOT / "combined_sentence_embedding_evaluation" / "best_threshold_metrics.csv"
    ).to_string(index=False))
    print(f"\n  Validation report: {VALIDATION_DIR / 'step7_sentence_embedding_validation.json'}")
    print("  [OK] Sentence-embedding evaluation complete.")


if __name__ == "__main__":
    main()
