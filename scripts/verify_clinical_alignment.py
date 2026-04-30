#!/usr/bin/env python3
"""
DEPRECATED TEST SCRIPT — Specific to one patient case (32cc reference).
This script is NOT part of the main pipeline and should not be used for general validation.
For proper validation, use scripts/5_validate_segmentation.py instead.
"""

import os
import sys
import json
from pathlib import Path


def verify_session():
    session_path = os.environ.get("PYBRAIN_SESSION")
    if not session_path:
        print("❌ Error: PYBRAIN_SESSION environment variable not set.")
        sys.exit(1)

    session_dir = Path(str(session_path)).parent
    stats_path = session_dir / "tumor_stats.json"
    morph_path = session_dir / "morphology.json"

    if not stats_path.exists():
        print(f"❌ Error: {stats_path} not found. Run Stage 3 first.")
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    v_whole = stats.get("volume_cc", {}).get("whole_tumor", 0.0)
    v_edema = stats.get("volume_cc", {}).get("edema", 0.0)

    print("--- Clinical Alignment Verification ---")
    print(f"Session: {session_dir.name}")
    print("Target Volume: 32.0 cc")
    print(f"AI Volume:     {v_whole:.2f} cc")

    # 1. Volume Alignment (5% tolerance)
    diff = abs(v_whole - 32.0)
    if diff <= 1.6:  # 5% of 32
        print("✅ Volume alignment: PASS (within 5%)")
    else:
        print(f"⚠️ Volume alignment: WARN ({diff:.2f}cc diff)")

    # 2. Edema Shell Presence
    if v_edema > 15.0:
        print(f"✅ Edema shell recovery: PASS ({v_edema:.1f} cc detected)")
    else:
        print(f"❌ Edema shell recovery: FAIL ({v_edema:.1f} cc - check threshold spread)")

    # 3. Multifocality (requires Step 7 morphology)
    if morph_path.exists():
        with open(morph_path) as f:
            morph = json.load(f)
        # Check if secondary lesions were preserved (logic implied by combined whole_cc)
        if morph.get("volumes", {}).get("whole_cc", 0) > 25.0:
            print("✅ Morphological consistency: PASS")
        else:
            print("❌ Morphological consistency: FAIL (Volume discrepancy)")
    else:
        print("ℹ️ Step 7 Morphology not yet run. Skipping multifocality check.")


if __name__ == "__main__":
    verify_session()
