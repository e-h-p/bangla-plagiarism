# -*- coding: utf-8 -*-
"""Step 1 - derive the canonical base dataset from the untouched source workbook.

Reads ``BanglaPlagPair-1K_source_links_final.xlsx`` read-only and writes a
derived copy that adds the reference-compatible aliases (``pair_no``,
``variant_type``) and the binary ``Label`` column. ``text1`` and ``text2`` are
carried through byte-for-byte; row order is preserved exactly.
"""

from __future__ import annotations

import json

import pandas as pd

from pipeline_common import (
    BASE_DATASET_CSV,
    BASE_DATASET_XLSX,
    EXPECTED_ROWS,
    ORIGINAL_DATASET,
    VALIDATION_DIR,
    banner,
    build_base_frame,
    ensure_dirs,
    read_original_dataset,
    set_global_seed,
    sha256_of,
)


def validate(original: pd.DataFrame, base: pd.DataFrame) -> dict[str, object]:
    checks: dict[str, object] = {}

    checks["row_count"] = len(base)
    checks["row_count_ok"] = len(base) == EXPECTED_ROWS == len(original)

    checks["pair_id_unique"] = bool(base["pair_id"].is_unique)
    checks["pair_id_monotonic"] = bool(base["pair_id"].is_monotonic_increasing)
    checks["pair_id_range"] = [int(base["pair_id"].min()), int(base["pair_id"].max())]
    checks["pair_no_matches_pair_id"] = bool(base["pair_no"].equals(base["pair_id"]))

    # Row order: the i-th derived row must be the i-th original row.
    checks["row_order_preserved"] = bool(
        base["pair_id"].reset_index(drop=True).equals(original["pair_id"].reset_index(drop=True))
    )

    # Texts must be identical, character for character.
    checks["text1_identical"] = bool(base["text1"].astype(str).equals(original["text1"].astype(str)))
    checks["text2_identical"] = bool(base["text2"].astype(str).equals(original["text2"].astype(str)))

    # Labels.
    derived = base["variant_type"].str.strip().str.lower().ne("mismatch").astype(int)
    checks["label_matches_binary_label"] = bool(derived.equals(base["binary_label"].astype(int)))
    checks["label_matches_provided"] = bool(base["Label"].equals(base["binary_label"].astype(int)))
    checks["label_distribution"] = {
        str(k): int(v) for k, v in base["Label"].value_counts().sort_index().items()
    }
    checks["pair_type_distribution"] = {
        str(k): int(v) for k, v in base["pair_type"].value_counts().items()
    }

    # Source links.
    for column in ("source_link_text1", "source_link_text2"):
        checks[f"{column}_identical"] = bool(
            base[column].fillna("").astype(str).equals(original[column].fillna("").astype(str))
        )
        checks[f"{column}_non_null"] = int(base[column].notna().sum())

    # Text hygiene.
    checks["empty_text1"] = int((base["text1"].astype(str).str.strip() == "").sum())
    checks["empty_text2"] = int((base["text2"].astype(str).str.strip() == "").sum())
    checks["identical_text_pairs"] = int((base["text1"].astype(str) == base["text2"].astype(str)).sum())
    checks["duplicate_text_pairs"] = int(base.duplicated(subset=["text1", "text2"]).sum())

    checks["text1_char_len"] = {
        "min": int(base["text1"].astype(str).str.len().min()),
        "mean": round(float(base["text1"].astype(str).str.len().mean()), 3),
        "max": int(base["text1"].astype(str).str.len().max()),
    }
    checks["text2_char_len"] = {
        "min": int(base["text2"].astype(str).str.len().min()),
        "mean": round(float(base["text2"].astype(str).str.len().mean()), 3),
        "max": int(base["text2"].astype(str).str.len().max()),
    }

    return checks


def main() -> None:
    set_global_seed()
    ensure_dirs()
    banner("STEP 1 - build canonical base dataset")

    source_hash_before = sha256_of(ORIGINAL_DATASET)
    original = read_original_dataset()
    base = build_base_frame(original)

    checks = validate(original, base)
    failures = [key for key, value in checks.items() if key.endswith("_ok") or key.endswith("_identical")
                or key.endswith("_preserved") or key.startswith("label_matches")
                or key in {"pair_id_unique", "pair_id_monotonic", "pair_no_matches_pair_id"}]
    failed = [key for key in failures if checks[key] is False]
    if failed:
        raise RuntimeError(f"Base dataset validation failed: {failed}")

    base.to_excel(BASE_DATASET_XLSX, index=False, engine="openpyxl")
    base.to_csv(BASE_DATASET_CSV, index=False, encoding="utf-8-sig")

    source_hash_after = sha256_of(ORIGINAL_DATASET)
    if source_hash_before != source_hash_after:
        raise RuntimeError("The original dataset file was modified. Aborting.")

    report = {
        "original_dataset": str(ORIGINAL_DATASET),
        "original_sha256": source_hash_after,
        "original_unchanged": True,
        "base_dataset_xlsx": str(BASE_DATASET_XLSX),
        "base_dataset_csv": str(BASE_DATASET_CSV),
        "checks": checks,
    }
    out = VALIDATION_DIR / "step1_base_dataset_validation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  Rows                : {len(base)}")
    print(f"  Positives / Negatives: {checks['label_distribution']}")
    print(f"  pair_type counts    : {checks['pair_type_distribution']}")
    print(f"  Original SHA-256    : {source_hash_after}")
    print(f"  Base workbook       : {BASE_DATASET_XLSX}")
    print(f"  Validation report   : {out}")
    print("  [OK] All structural checks passed.")


if __name__ == "__main__":
    main()
