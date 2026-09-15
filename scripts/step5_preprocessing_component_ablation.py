# -*- coding: utf-8 -*-
"""Step 5 - component-wise preprocessing ablation on BanglaPlagPair-1K.

Runs the component-wise preprocessing ablation against the
updated dataset. Rather than re-implementing the ablation, the reference module
is imported and its path constants are redirected at run time, so the nine
profiles, the transformations, the metric definitions, the threshold grid and
the summary/figure logic are byte-identical to the previous round.

Profiles evaluated (each applied in isolation, plus the full bundle):
    raw, unicode_normalization, lowercasing, digit_normalization,
    punctuation_removal, lexical_mapping, stopword_removal, stemming,
    full_pipeline
"""

from __future__ import annotations

import json

import pandas as pd

from pipeline_common import (
    ABLATION_DIR,
    BASE_DATASET_XLSX,
    METRICS,
    VALIDATION_DIR,
    WORK_ROOT,
    banner,
    ensure_dirs,
    set_global_seed,
)

import run_preprocessing_component_ablation as ablation


FIGURES_DIR = ABLATION_DIR / "figures"


def redirect_reference_paths() -> None:
    """Point the reference ablation module at this working directory."""
    ablation.SCRIPT_DIR = WORK_ROOT
    ablation.INPUT_FILE = BASE_DATASET_XLSX
    ablation.OUTPUT_DIR = ABLATION_DIR
    ablation.FIGURES_DIR = FIGURES_DIR


def validate_outputs() -> dict[str, object]:
    """Confirm every profile evaluated the same 1000 rows with the same labels."""
    base = pd.read_excel(BASE_DATASET_XLSX, engine="openpyxl")
    expected_labels = base["Label"].astype(int).tolist()
    expected_ids = base["pair_no"].tolist()

    profiles = [p.slug for p in ablation.build_profiles()]
    per_profile: dict[str, object] = {}

    for slug in profiles:
        labeled = pd.read_csv(ABLATION_DIR / "evaluations" / slug / "labeled_scores.csv")
        if len(labeled) != len(base):
            raise RuntimeError(f"[{slug}] row count changed: {len(labeled)}")
        if labeled["pair_no"].tolist() != expected_ids:
            raise RuntimeError(f"[{slug}] pair_no order changed.")
        if labeled["Label"].astype(int).tolist() != expected_labels:
            raise RuntimeError(f"[{slug}] labels changed.")
        for metric in METRICS:
            if labeled[metric].isna().any():
                raise RuntimeError(f"[{slug}] metric '{metric}' has NaN values.")
            if float(labeled[metric].min()) < 0 or float(labeled[metric].max()) > 1:
                raise RuntimeError(f"[{slug}] metric '{metric}' out of [0, 1].")
        per_profile[slug] = {
            "rows": int(len(labeled)),
            "positives": int(labeled["Label"].sum()),
            "mean_scores": {m: round(float(labeled[m].mean()), 6) for m in METRICS},
        }

    return {"profiles": profiles, "per_profile": per_profile}


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 5 - component-wise preprocessing ablation")

    redirect_reference_paths()
    print(f"  Input        : {ablation.INPUT_FILE}")
    print(f"  Output       : {ablation.OUTPUT_DIR}")
    print(f"  Figures      : {ablation.FIGURES_DIR}")
    print(f"  Profiles     : {[p.slug for p in ablation.build_profiles()]}")
    print(f"  Fixed t      : {ablation.FIXED_THRESHOLD}")
    print(f"  Sweep        : {ablation.THRESHOLDS}\n")

    ablation.main()

    checks = validate_outputs()
    (VALIDATION_DIR / "step5_ablation_validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fixed = pd.read_csv(ABLATION_DIR / "fixed_threshold_summary.csv")
    print("\n  --- Fixed-threshold F1 by profile (t=0.50) ---")
    pivot = fixed.pivot(index="Profile", columns="Metric", values="F1")
    pivot = pivot.reindex([p.slug for p in ablation.build_profiles()])[METRICS]
    print(pivot.round(4).to_string())

    print(f"\n  Validation report: {VALIDATION_DIR / 'step5_ablation_validation.json'}")
    print("  [OK] Ablation complete.")


if __name__ == "__main__":
    main()
