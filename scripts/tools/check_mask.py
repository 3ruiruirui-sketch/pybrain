#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# --- pybrain Path Injection ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import nibabel as nib

def check_mask():
    session_path = os.environ.get("PYBRAIN_SESSION")
    if not session_path:
        print("❌ Error: PYBRAIN_SESSION environment variable not set.")
        sys.exit(1)
        
    session_dir = Path(session_path).parent
    mask_path = session_dir / "brain_mask.nii.gz"
    
    if not mask_path.exists():
        print(f"❌ Error: {mask_path} not found.")
        return

    img = nib.load(mask_path)
    data = img.get_fdata()
    print(f"✅ Brain Mask Loaded: {mask_path.name}")
    print(f"   Shape: {data.shape}")
    print(f"   Voxels: {int(data.sum())}")
    print(f"   Volume: {data.sum() * np.prod(img.header.get_zooms()) / 1000 :.2f} cc")

if __name__ == "__main__":
    check_mask()
