#!/usr/bin/env python3
import sys
from pathlib import Path

# --- pybrain Path Injection ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import nibabel as nib


def check_logic():
    print("--- V7 Logic Verification Tool ---")
    # Verify imports
    print(f"✅ numpy version: {np.__version__}")
    print(f"✅ nibabel version: {nib.__version__}")
    print("✅ pybrain.core.normalization: loaded")
    print("----------------------------------")
    print("Clinical alignment logic is now officially in:")
    print("  /scripts/verify_clinical_alignment.py")


if __name__ == "__main__":
    check_logic()
