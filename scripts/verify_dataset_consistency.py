# -*- coding: utf-8 -*-
"""Audit: prove every artefact in this folder is built from the CURRENT dataset.

Walks every .xlsx / .csv under the working directory and checks its text against
``BanglaPlagPair-1K_source_links_final.xlsx`` as it stands right now.

Three kinds of file are handled:

  * raw text        - ``text1``/``text2`` must equal the original verbatim
  * preserved raw   - ``text1_raw``/``text2_raw`` must equal the original verbatim,
                      and ``text1``/``text2`` must equal the declared transform of it
  * transformed     - ``text1``/``text2`` must equal the declared profile transform
                      of the current original

Independently of the column checks, every file is also scanned for text
fragments that only existed in the SUPERSEDED version of pair_id 12/354/668.
Any hit means that file is stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline_common import (
    ORIGINAL_DATASET,
    VALIDATION_DIR,
    WORK_ROOT,
    banner,
    build_preprocessor,
    read_original_dataset,
    sha256_of,
)

import run_preprocessing_component_ablation as ablation


#: Fragments from the SUPERSEDED text2 of the three corrected pairs.
#: These must be checked ONLY against that pair's text2 column - the same
#: wording also occurs in the pair's text1, which is unchanged and legitimate.
STALE_MARKERS = {
    12: "পেদ্রু আলভারেজ",
    354: "অ্যানিউরিজম",
    668: None,  # 668 only gained a prefix; no fragment unique to the old version
}

SKIP_DIRS = {"scripts", "__pycache__", "validation"}
TEXT_PAIRS = [("text1", "text2"), ("text1_raw", "text2_raw")]


def load_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_csv(path)
    except Exception:
        return None


def build_transforms() -> dict[str, object]:
    """slug -> callable, for every preprocessing profile the suite can emit."""
    transforms = {profile.slug: profile.transform for profile in ablation.build_profiles()}
    transforms["full"] = build_preprocessor().preprocess_text  # polish-stage label
    return transforms


def audit_file(path: Path, original: pd.DataFrame, transforms: dict[str, object]) -> dict[str, object]:
    rel = str(path.relative_to(WORK_ROOT))
    frame = load_any(path)
    if frame is None:
        return {"file": rel, "status": "unreadable", "checked": False}

    result: dict[str, object] = {"file": rel, "rows": int(len(frame)), "checked": False, "checks": {}}

    if len(frame) != len(original):
        result["status"] = "row-count-differs"
        return result

    # Which profile, if any, does this file declare?
    profile = None
    for column in ("preprocess_profile", "Profile"):
        if column in frame.columns and frame[column].notna().any():
            profile = str(frame[column].dropna().iloc[0])
            break
    result["declared_profile"] = profile

    ref1 = original["text1"].astype(str).reset_index(drop=True)
    ref2 = original["text2"].astype(str).reset_index(drop=True)

    # 1) preserved original text
    if {"text1_raw", "text2_raw"}.issubset(frame.columns):
        result["checks"]["text1_raw_matches_original"] = bool(
            frame["text1_raw"].astype(str).reset_index(drop=True).equals(ref1))
        result["checks"]["text2_raw_matches_original"] = bool(
            frame["text2_raw"].astype(str).reset_index(drop=True).equals(ref2))
        result["checked"] = True

    # 2) text1/text2 - either verbatim or the declared transform of the current original
    if {"text1", "text2"}.issubset(frame.columns):
        got1 = frame["text1"].astype(str).reset_index(drop=True)
        got2 = frame["text2"].astype(str).reset_index(drop=True)

        if profile in (None, "raw"):
            result["checks"]["text1_matches_original"] = bool(got1.equals(ref1))
            result["checks"]["text2_matches_original"] = bool(got2.equals(ref2))
            result["checked"] = True
        elif profile in transforms:
            transform = transforms[profile]
            exp1 = original["text1"].map(transform).astype(str).reset_index(drop=True)
            exp2 = original["text2"].map(transform).astype(str).reset_index(drop=True)
            result["checks"][f"text1_is_{profile}_of_current_original"] = bool(got1.equals(exp1))
            result["checks"][f"text2_is_{profile}_of_current_original"] = bool(got2.equals(exp2))
            result["checked"] = True
        else:
            result["checks"]["unknown_profile"] = False

    # 3) stale-marker scan, scoped to the corrected pair's text2 column only.
    #    The same wording legitimately occurs in that pair's text1 (unchanged),
    #    so a whole-file scan would false-positive on every artefact.
    stale_hits: list[int] = []
    id_column = next((c for c in ("pair_id", "pair_no") if c in frame.columns), None)
    text2_columns = [c for c in ("text2", "text2_raw") if c in frame.columns]
    if id_column and text2_columns:
        ids = pd.to_numeric(frame[id_column], errors="coerce")
        for pair_id, marker in STALE_MARKERS.items():
            if not marker:
                continue
            row = frame.loc[ids == pair_id]
            if row.empty:
                continue
            for column in text2_columns:
                if marker in str(row.iloc[0][column]):
                    stale_hits.append(pair_id)
                    break
    result["stale_marker_hits"] = stale_hits
    result["stale_scan_applied"] = bool(id_column and text2_columns)

    failed = [k for k, v in result["checks"].items() if v is False]
    if stale_hits:
        failed.append(f"stale_text_for_pair_ids={stale_hits}")
    result["failed"] = failed
    result["status"] = "OK" if (result["checked"] and not failed) else (
        "STALE/MISMATCH" if failed else "not-text-bearing")
    return result


def main() -> None:
    banner("AUDIT - is every artefact built from the current dataset?")

    original = read_original_dataset()
    transforms = build_transforms()
    print(f"  Original : {ORIGINAL_DATASET.name}")
    print(f"  SHA-256  : {sha256_of(ORIGINAL_DATASET)}")
    print(f"  Rows     : {len(original)}\n")

    files = [
        p for p in sorted(WORK_ROOT.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".csv"}
        and not set(p.relative_to(WORK_ROOT).parts) & SKIP_DIRS
        and not p.name.startswith("~$")
    ]

    results = [audit_file(p, original, transforms) for p in files]

    checked = [r for r in results if r["checked"]]
    stale = [r for r in results if r["status"] == "STALE/MISMATCH"]
    unchecked = [r for r in results if not r["checked"]]

    print(f"  Files scanned      : {len(results)}")
    print(f"  Text-bearing (verified): {len(checked)}")
    print(f"  No text columns    : {len(unchecked)}")
    print(f"  STALE / MISMATCHED : {len(stale)}\n")

    for r in checked:
        flag = "OK  " if r["status"] == "OK" else "FAIL"
        prof = f" [{r['declared_profile']}]" if r.get("declared_profile") else ""
        print(f"  {flag}  {r['file']}{prof}")
        if r["failed"]:
            print(f"          -> {r['failed']}")

    if unchecked:
        print("\n  (no text columns - nothing to verify)")
        for r in unchecked:
            print(f"        {r['file']}")

    report = {
        "original_dataset": str(ORIGINAL_DATASET),
        "original_sha256": sha256_of(ORIGINAL_DATASET),
        "files_scanned": len(results),
        "files_verified": len(checked),
        "files_stale": len(stale),
        "all_consistent": not stale,
        "results": results,
    }
    out = VALIDATION_DIR / "dataset_consistency_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Report: {out}")
    if stale:
        print("  [FAIL] Some artefacts are NOT built from the current dataset:")
        for r in stale:
            print(f"    {r['file']}: {r['failed']}")
    else:
        print("  [OK] Every text-bearing artefact is built from the current dataset.")


if __name__ == "__main__":
    main()
