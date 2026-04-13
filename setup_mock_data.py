import os
import numpy as np
import nibabel as nib
from pathlib import Path

# Create data directory
Path("data").mkdir(exist_ok=True)

# Define mock data dimensions
shape = (10, 10, 10)
affine = np.eye(4)

# Create train_manifest.csv
manifest_content = """patient_id,image_path,mask_path,target,split
UPENN-GBM-0001,./data/UPENN-GBM-0001_t1ce.nii.gz,./data/UPENN-GBM-0001_mask.nii.gz,1,train
UPENN-GBM-0002,./data/UPENN-GBM-0002_t1ce.nii.gz,./data/UPENN-GBM-0002_mask.nii.gz,0,train
UPENN-GBM-0003,./data/UPENN-GBM-0003_t1ce.nii.gz,./data/UPENN-GBM-0003_mask.nii.gz,1,val
"""
with open("train_manifest.csv", "w") as f:
    f.write(manifest_content)

# Generate mock NIfTI files
patients = ["UPENN-GBM-0001", "UPENN-GBM-0002", "UPENN-GBM-0003"]
for p in patients:
    img = nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), affine)
    mask = nib.Nifti1Image((np.random.rand(*shape) > 0.5).astype(np.uint8), affine)
    nib.save(img, f"data/{p}_t1ce.nii.gz")
    nib.save(mask, f"data/{p}_mask.nii.gz")

print("Created train_manifest.csv and mock NIfTI files.")
