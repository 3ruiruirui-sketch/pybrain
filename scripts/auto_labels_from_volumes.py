#!/usr/bin/env python3
"""
Auto-Generate Approximate Grade Labels from BraTS Volumes
=========================================================
Heuristic-based grade estimation using tumor volume characteristics.

APPROXIMATE GRADING RULES (from clinical literature):
  - Grade II (LGG):  WT < 40cc, minimal/no ET, no necrosis
  - Grade III (AA):  WT 30-80cc, moderate ET, some necrosis
  - Grade IV (GBM):  WT > 60cc, prominent ET (ring enhancement), necrosis

BraTS Label mapping:
  WT = labels 1+2+4 (all tumor)
  TC = label 1 (necrotic/core)
  ET = label 4 (enhancing)

OUTPUT:
  data/auto_generated_labels.csv — case_id, grade, grade_approx

⚠️  DISCLAIMER: These are APPROXIMATE grades based on imaging heuristics.
    NOT ground truth. For research/development only.
    Do NOT use for clinical decisions.
"""

import csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRATS_DATA = PROJECT_ROOT / "data/datasets/BraTS2021/raw/BraTS2021_Training_Data"
OUTPUT_CSV = PROJECT_ROOT / "data" / "auto_generated_labels.csv"


def get_tumor_volumes(case_id: str) -> dict:
    """Extract WT, TC, ET volumes from BraTS segmentation."""
    import nibabel as nib

    case_dir = BRATS_DATA / case_id
    seg_file = case_dir / f"{case_id}_seg.nii.gz"

    if not seg_file.exists():
        return None

    try:
        img = nib.load(str(seg_file), mmap=True)
        data = img.get_fdata()
        zooms = img.header.get_zooms()[:3]
        vox_vol = np.prod(zooms) / 1000.0  # cc

        # BraTS2021: labels 0=BG, 1=NCR/NET, 2=ED, 4=ET
        wt = ((data == 1) | (data == 2) | (data == 4)).sum() * vox_vol
        tc = (data == 1).sum() * vox_vol  # Necrotic/Core
        et = (data == 4).sum() * vox_vol  # Enhancing
        ed = (data == 2).sum() * vox_vol  # Edema
        nc = (tc > 0).sum()  # Has necrotic component

        return {"wt": wt, "tc": tc, "et": et, "ed": ed, "nc": nc}

    except Exception:
        return None


def estimate_grade(vol: dict) -> tuple:
    """
    Estimate WHO grade from tumor volumes.

    Heuristics (from BraTS clinical literature):
      Grade II (LGG):  WT < 40cc, ET/TC ratio low, no ring enhancement
      Grade III (AA):  WT 40-80cc, moderate ET, some necrosis
      Grade IV (GBM):  WT > 60cc, high ET (ring), significant necrosis

    Returns: (grade_str, is_lgg, confidence)
    """
    wt = vol.get("wt", 0)
    et = vol.get("et", 0)
    tc = vol.get("tc", 0)
    vol.get("ed", 0)
    vol.get("nc", 0)

    et_ratio = et / wt if wt > 0 else 0
    tc_ratio = tc / wt if wt > 0 else 0

    # Grade IV (GBM) — large tumors, prominent ET, necrosis
    if wt > 60 and et_ratio > 0.15 and tc_ratio > 0.05:
        return "IV", 0, "high"
    # Grade IV (GBM) — ring enhancement pattern (high ET, necrotic core)
    elif wt > 50 and et > 10 and tc > 5:
        return "IV", 0, "high"
    # Grade III (AA) — intermediate
    elif wt > 35 and (et_ratio > 0.05 or tc_ratio > 0.05):
        return "III", 0, "medium"
    # Grade II (LGG) — small, no/minimal enhancement
    elif wt > 0 and wt < 40:
        if et_ratio < 0.05 and tc_ratio < 0.10:
            return "II", 1, "medium"
        else:
            return "II", 1, "low"  # uncertain
    # Very small tumors — likely LGG
    elif wt > 0 and wt < 20:
        return "II", 1, "low"
    else:
        return "III", 0, "low"  # default to HGG if uncertain


def main():
    print("=" * 60)
    print("  Auto-Generating Grade Labels from BraTS Volumes")
    print("=" * 60)

    # Find all BraTS cases
    cases = sorted([d.name for d in BRATS_DATA.iterdir() if d.is_dir() and d.name.startswith("BraTS2021_")])
    print(f"\n  Found {len(cases)} BraTS2021 cases")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    results = []
    grade_dist = {"II": 0, "III": 0, "IV": 0, "unknown": 0}
    confidence_dist = {"high": 0, "medium": 0, "low": 0}

    print("\n  Processing cases...")
    for i, case_id in enumerate(cases, 1):
        if i % 100 == 0:
            print(f"    {i}/{len(cases)}...")

        vol = get_tumor_volumes(case_id)
        if vol is None:
            results.append(
                {
                    "case_id": case_id,
                    "grade": "unknown",
                    "grade_approx": "HGG",
                    "confidence": "none",
                    "wt": 0,
                    "tc": 0,
                    "et": 0,
                    "ed": 0,
                }
            )
            grade_dist["unknown"] += 1
            continue

        grade, is_lgg, confidence = estimate_grade(vol)
        grade_approx = "LGG" if is_lgg else "HGG"

        results.append(
            {
                "case_id": case_id,
                "grade": grade,
                "grade_approx": grade_approx,
                "confidence": confidence,
                "wt": round(vol["wt"], 2),
                "tc": round(vol["tc"], 2),
                "et": round(vol["et"], 2),
                "ed": round(vol["ed"], 2),
            }
        )

        grade_dist[grade] = grade_dist.get(grade, 0) + 1
        confidence_dist[confidence] += 1

    # Save CSV
    fieldnames = ["case_id", "grade", "grade_approx", "confidence", "wt", "tc", "et", "ed"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  Saved: {OUTPUT_CSV}")
    print("\n  Grade Distribution:")
    print(f"    Grade II (LGG): {grade_dist['II']}")
    print(f"    Grade III (AA): {grade_dist['III']}")
    print(f"    Grade IV (GBM): {grade_dist['IV']}")
    print(f"    Unknown:         {grade_dist['unknown']}")

    print("\n  Confidence:")
    print(f"    High:   {confidence_dist['high']}")
    print(f"    Medium: {confidence_dist['medium']}")
    print(f"    Low:    {confidence_dist['low']}")

    n_labeled = grade_dist["II"] + grade_dist["III"] + grade_dist["IV"]
    print(f"\n  Total labeled: {n_labeled}/{len(cases)}")

    print("\n  ⚠️  DISCLAIMER:")
    print("     These grades are APPROXIMATE based on imaging heuristics.")
    print("     NOT ground truth histology. For research/development only.")
    print("     Do NOT use for clinical decisions.")

    return results


if __name__ == "__main__":
    main()
