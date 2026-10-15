# -*- coding: utf-8 -*-
"""Step 9 - P1-P8 prompt-engineering evaluation on BanglaPlagPair-1K.

Evaluates the eight engineered prompts over the three selected open-weight
models, using the same protocol as every other stage of the suite
(implemented locally in ``evaluation_common.py``): binary labels from ``pair_type``,
fixed threshold 0.50, threshold sweep 0.10-0.90, best threshold chosen by
F1 -> Accuracy -> Recall -> Precision.

Models:
    Gemma 2 9B IT, BanglaLLama 3 8B Instruct, Llama 3 8B Bangla 4bit

Prompts:
    P1 Few-shot anchored          P5 NLI semantic
    P2 Contradiction aware        P6 Multidimensional
    P3 Hard-negative aware        P7 Calibrated conservative
    P4 Bangla instruction *       P8 Direct short

    * P4 replaces the previous round's "code-mixed bilingual" prompt.

Outputs (mirroring the layout used by the other evaluation stages):
    prompt_engineering_evaluation/
      evaluations/<model_slug>_<prompt>/   per-configuration evaluation
      fixed_threshold_summary.csv          all 24 configs at t=0.50
      best_threshold_summary.csv           all 24 configs, calibrated
      prompt_strategy_averages.csv         per-prompt means across models
      threshold_sweep_summary.csv
      prompt_engineering_summary.xlsx
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from pipeline_common import (
    BASE_DATASET_XLSX,
    FIXED_THRESHOLD,
    THRESHOLDS,
    VALIDATION_DIR,
    WORK_ROOT,
    banner,
    best_thresholds,
    classification_metrics,
    ensure_dirs,
    evaluate_scores,
    set_global_seed,
    threshold_sweep,
    write_evaluation_outputs,
)


INPUT_DIR = WORK_ROOT / "8 prompting results"
OUTPUT_DIR = WORK_ROOT / "prompt_engineering_evaluation"

MODELS = [
    ("Gemma 2 9B IT", "gemma_2_9b_it", "Gemma_2_9B_IT",
     "combined_1000_testcases Gemma_2_9B_IT .xlsx"),
    ("BanglaLLama 3 8B Instruct", "banglallama_3_8b_instruct", "BanglaLLama_3_8B_Instruct",
     "combined_1000_testcases BanglaLLama_3_8B_Instruct.xlsx"),
    ("Llama 3 8B Bangla 4bit", "llama_3_8b_bangla_4bit", "Llama_3_8B_Bangla_4bit",
     "combined_1000_testcases Llama_3_8B_Bangla_4bit.xlsx"),
]

#: Prompt identifier -> (short code, report label). Order is the report order.
PROMPTS = [
    ("P1_FewShot_Anchored", "P1", "Few-shot anchored"),
    ("P2_Contradiction_Aware", "P2", "Contradiction aware"),
    ("P3_Hard_Negative_Aware", "P3", "Hard-negative aware"),
    ("P4_Bangla_Instruction", "P4", "Bangla instruction"),
    ("P5_NLI_Semantic", "P5", "NLI semantic"),
    ("P6_MultiDimensional", "P6", "Multidimensional"),
    ("P7_Calibrated_Conservative", "P7", "Calibrated conservative"),
    ("P8_Direct_Short", "P8", "Direct short"),
]


def load_base() -> pd.DataFrame:
    base = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")
    return base[["pair_id", "pair_no", "variant_type", "pair_type", "Label", "text1", "text2"]].copy()


def load_model_scores(base: pd.DataFrame, filename: str, prefix: str) -> tuple[pd.DataFrame, dict]:
    path = INPUT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing prompting workbook: {path}")
    df = pd.read_excel(path, engine="openpyxl")

    if not df["pair_id"].is_unique:
        raise RuntimeError(f"{filename}: pair_id is not unique")
    if df["pair_id"].tolist() != base["pair_id"].tolist():
        raise RuntimeError(f"{filename}: pair_id order differs from the base dataset")

    # Provenance: does this run use the CURRENT text, or a superseded copy?
    text1_ok = bool(df["text1"].astype(str).equals(base["text1"].astype(str)))
    text2_ok = bool(df["text2"].astype(str).equals(base["text2"].astype(str)))
    stale_ids = df.loc[df["text2"].astype(str).ne(base["text2"].astype(str)), "pair_id"].tolist()

    if not (text1_ok and text2_ok):
        raise RuntimeError(f"{filename}: input texts differ from the canonical dataset")

    columns = {}
    for key, code, _label in PROMPTS:
        column = f"{prefix}_{key}_Score"
        if column not in df.columns:
            raise RuntimeError(f"{filename}: missing column {column}")
        columns[code] = column

    merged = base.copy()
    for code, column in columns.items():
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isna().any():
            raise RuntimeError(f"{filename}: {column} contains non-numeric values")
        if not series.between(0.0, 1.0).all():
            raise RuntimeError(f"{filename}: {column} contains out-of-range scores")
        merged[code] = series.values

    provenance = {
        "file": filename,
        "text1_matches_current_dataset": text1_ok,
        "text2_matches_current_dataset": text2_ok,
        "stale_pair_ids": stale_ids,
    }
    return merged, provenance


def tag(frame: pd.DataFrame, model: str, slug: str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "Model", model)
    out.insert(1, "ModelSlug", slug)
    return out


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 9 - P1-P8 prompt-engineering evaluation")

    base = load_base()
    print(f"  Base dataset : {BASE_DATASET_XLSX.name} ({len(base)} pairs, "
          f"{int(base['Label'].sum())} positive / {int((base['Label'] == 0).sum())} mismatch)")
    print(f"  Input dir    : {INPUT_DIR}")
    print(f"  Prompts      : {[c for _, c, _ in PROMPTS]}")
    print(f"  Fixed t      : {FIXED_THRESHOLD} | sweep {THRESHOLDS[0]}-{THRESHOLDS[-1]}\n")

    label_of = {code: label for _, code, label in PROMPTS}
    fixed_parts, best_parts, sweep_parts = [], [], []
    provenance_all = []

    for model, slug, prefix, filename in MODELS:
        scored, provenance = load_model_scores(base, filename, prefix)
        provenance_all.append({"model": model, **provenance})

        flag = "" if provenance["text2_matches_current_dataset"] else \
            f"  [WARNING] scored on superseded text2 for pair_id {provenance['stale_pair_ids']}"
        print(f"  {model}{flag}")

        codes = [c for _, c, _ in PROMPTS]
        labeled, class_df, sweep_df, best_df = evaluate_scores(scored, codes)

        for frame in (class_df, sweep_df, best_df):
            frame["PromptLabel"] = frame["Metric"].map(label_of)

        fixed_parts.append(tag(class_df, model, slug))
        best_parts.append(tag(best_df, model, slug))
        sweep_parts.append(tag(sweep_df, model, slug))

        # Per-configuration evaluation folders, same file set as every other stage.
        for code in codes:
            single = scored[["pair_id", "pair_no", "variant_type", "pair_type", "Label", code]].copy()
            out_dir = OUTPUT_DIR / "evaluations" / f"{slug}_{code}"
            lab, cls, swp, bst = evaluate_scores(single, [code])
            write_evaluation_outputs(
                output_dir=out_dir,
                labeled_scores=lab,
                class_df=cls,
                sweep_df=swp,
                best_df=bst,
                metrics=[code],
                interactive_filename=f"interactive_threshold_{slug}_{code}.xlsx",
            )

        row = class_df.set_index("Metric")
        for code in codes:
            print(f"     {code} {label_of[code]:24s} F1={row.loc[code, 'F1']:.4f} "
                  f"Spec={row.loc[code, 'Specificity']:.4f} FP={int(row.loc[code, 'FP'])} "
                  f"FN={int(row.loc[code, 'FN'])}")
        print()

    fixed = pd.concat(fixed_parts, ignore_index=True)
    best = pd.concat(best_parts, ignore_index=True)
    sweep = pd.concat(sweep_parts, ignore_index=True)

    # Prompt-strategy averages across the three models (report Table "prompt-average").
    averages = (
        fixed.groupby(["Metric", "PromptLabel"], sort=False)
        .agg(Avg_F1=("F1", "mean"), Avg_Recall=("Recall", "mean"),
             Avg_Specificity=("Specificity", "mean"),
             Avg_BalancedAccuracy=("BalancedAccuracy", "mean"),
             Avg_Accuracy=("Accuracy", "mean"))
        .round(6).reset_index().rename(columns={"Metric": "Prompt"})
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(OUTPUT_DIR / "fixed_threshold_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUTPUT_DIR / "best_threshold_summary.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(OUTPUT_DIR / "threshold_sweep_summary.csv", index=False, encoding="utf-8-sig")
    averages.to_csv(OUTPUT_DIR / "prompt_strategy_averages.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUTPUT_DIR / "prompt_engineering_summary.xlsx", engine="openpyxl") as writer:
        fixed.to_excel(writer, sheet_name="fixed_threshold_0_50", index=False)
        best.to_excel(writer, sheet_name="best_threshold", index=False)
        averages.to_excel(writer, sheet_name="prompt_averages", index=False)
        sweep.to_excel(writer, sheet_name="threshold_sweep", index=False)

    (VALIDATION_DIR / "step9_prompt_engineering_validation.json").write_text(
        json.dumps({
            "rows": int(len(base)),
            "models": [m for m, _, _, _ in MODELS],
            "prompts": [{"code": c, "key": k, "label": l} for k, c, l in PROMPTS],
            "configurations": len(MODELS) * len(PROMPTS),
            "fixed_threshold": FIXED_THRESHOLD,
            "threshold_sweep": THRESHOLDS,
            "provenance": provenance_all,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("  --- Top fixed-threshold configurations (t=0.50) ---")
    top = fixed.sort_values(["F1", "Specificity"], ascending=False).head(10)
    print(top[["Model", "Metric", "PromptLabel", "F1", "Accuracy", "Recall",
               "Specificity", "FP", "FN"]].round(4).to_string(index=False))

    print("\n  --- Prompt-strategy averages across the three models ---")
    print(averages.round(4).to_string(index=False))

    print(f"\n  Outputs: {OUTPUT_DIR}")
    print("  [OK] Prompt-engineering evaluation complete.")


if __name__ == "__main__":
    main()
