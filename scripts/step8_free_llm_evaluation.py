# -*- coding: utf-8 -*-
"""Step 8 - free Hugging Face LLM evaluation on BanglaPlagPair-1K.

Evaluates the archived open-weight LLM score workbook
supplied for the updated dataset. The reference module is imported and its path
constants are redirected at run time, so the per-model evaluation, the combined
report, the score normalization/clipping and the output layout are identical to
the previous round.

Models evaluated (prompted scores supplied in the input workbook):
    Qwen2.5 0.5B Instruct, Qwen2.5 1.5B Instruct, Gemma 2 2B IT, Gemma 2 9B IT,
    Aya Expanse 8B, Llama 3.2 3B Instruct, Llama 3 8B Bangla 4bit,
    BanglaLLama 3 8B Instruct

Two adaptations, neither of which changes the methodology:
  * all eight models live in ONE workbook here (the old round had one workbook
    per model), so every config points at the same ``input_file``;
  * rows are keyed on ``pair_id`` instead of ``pair_no`` - the values are
    identical, and ``normalize_model_scores`` maps it back onto ``pair_no``.
"""

from __future__ import annotations

import json
import sys
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

import evaluate_free_hf_models as free_llm


OUTPUT_ROOT = WORK_ROOT / "free llm evualation"
# Rescored after the text2 corrections to pair_id 12, 354 and 668.
# (previous input: "combined_1000_testcases (1).xlsx", scored on the old text2)
INPUT_FILE = OUTPUT_ROOT / "combined_1000_testcases2.xlsx"


def redirect_reference_paths() -> None:
    """Point the reference module at this working directory and input workbook."""
    free_llm.SCRIPT_DIR = WORK_ROOT
    free_llm.BASE_TESTCASES = BASE_DATASET_XLSX
    free_llm.OUTPUT_ROOT = OUTPUT_ROOT
    free_llm.COMBINED_OUTPUT_DIR = OUTPUT_ROOT / "combined_free_hf_model_evaluation"
    # All eight models now share one workbook and key on pair_id.
    free_llm.MODELS = [
        replace(config, input_file=INPUT_FILE, id_column="pair_id")
        for config in free_llm.MODELS
    ]


def preflight() -> dict[str, object]:
    """Confirm the supplied workbook lines up with the untouched base dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input workbook: {INPUT_FILE}")

    scores = pd.read_excel(INPUT_FILE, engine="openpyxl")
    base = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")

    source_columns: list[str] = []
    for config in free_llm.MODELS:
        source_columns.extend(config.score_columns)
    missing = [c for c in source_columns if c not in scores.columns]
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
        "models": [
            {
                "model_name": config.model_name,
                "slug": config.slug,
                "source_column": config.score_columns[0],
                "metric_column": config.metric_column,
                "min": round(float(scores[config.score_columns[0]].min()), 6),
                "max": round(float(scores[config.score_columns[0]].max()), 6),
                "mean": round(float(scores[config.score_columns[0]].mean()), 6),
                "missing": int(scores[config.score_columns[0]].isna().sum()),
                "below_0": int((scores[config.score_columns[0]] < 0).sum()),
                "above_1": int((scores[config.score_columns[0]] > 1).sum()),
                "distinct_values": int(scores[config.score_columns[0]].nunique()),
            }
            for config in free_llm.MODELS
        ],
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
    directories = [f"{config.slug}_evaluation" for config in free_llm.MODELS]
    directories.append("combined_free_hf_model_evaluation")

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
    banner("STEP 8 - free Hugging Face LLM evaluation")

    redirect_reference_paths()
    print(f"  Base dataset : {free_llm.BASE_TESTCASES}")
    print(f"  Input scores : {INPUT_FILE}")
    print(f"  Output root  : {free_llm.OUTPUT_ROOT}")
    print(f"  Models       : {len(free_llm.MODELS)}\n")

    checks = preflight()
    print("  Preflight: input workbook matches the base dataset (ids, order, texts).")
    for info in checks["models"]:
        print(f"    {info['source_column']:30s} [{info['min']:.4f}, {info['max']:.4f}] "
              f"mean={info['mean']:.4f}  distinct={info['distinct_values']}")
    print()

    # The reference main() parses sys.argv; keep it clean so it takes the defaults.
    saved_argv = sys.argv
    sys.argv = ["evaluate_free_hf_models"]
    try:
        free_llm.main()
    finally:
        sys.argv = saved_argv

    checks["per_output"] = postflight()
    (VALIDATION_DIR / "step8_free_llm_validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    combined = OUTPUT_ROOT / "combined_free_hf_model_evaluation"
    print("\n  --- Combined evaluation @ t=0.50 ---")
    print(pd.read_csv(combined / "classification_metrics_threshold_0_50.csv").to_string(index=False))
    print("\n  --- Best threshold per model ---")
    print(pd.read_csv(combined / "best_threshold_metrics.csv").to_string(index=False))
    print(f"\n  Validation report: {VALIDATION_DIR / 'step8_free_llm_validation.json'}")
    print("  [OK] Free LLM evaluation complete.")


if __name__ == "__main__":
    main()
