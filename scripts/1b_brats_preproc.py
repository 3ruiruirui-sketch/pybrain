#!/usr/bin/env python3
"""
Stage 1b — BraTS Pre-processing (SimpleITK + Multi-Contrast)
===========================================================
Applies LPS orientation (scanner-native, DICOM standard),
1mm³ isotropic resampling, and multi-contrast brain extraction.
"""

import sys
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pybrain.io.session import get_session, get_paths
from pybrain.core.brainmask import robust_brain_mask
from pybrain.core.metrics import compute_volume_cc
from pybrain.io.nifti_io import save_nifti
from pybrain.io.logging_utils import setup_logging

def reorient_resample(in_path, out_path, is_label=False):
    """Apply Orientation and Spacing via SimpleITK."""
    if not in_path.exists():
        return None

    # B1 fix: Use strict '>' to avoid skipping when input and output have identical timestamps
    # (can occur with git clone, shared volumes, or FAT32 filesystems).
    # The original '>=' treats equal timestamps as "already done" which is incorrect —
    # equal timestamps mean the output was produced at the same second as the input,
    # not that the input is older than the output.
    if out_path.exists() and out_path.stat().st_size > 10_000:
        m_in  = in_path.stat().st_mtime
        m_out = out_path.stat().st_mtime
        if m_out > m_in:
            return out_path   # output is genuinely newer — skip

    itk_img = sitk.ReadImage(str(in_path))

    # 1. Handle 4D images (e.g. multi-echo) — extract first 3D volume
    if itk_img.GetDimension() > 3:
        size = list(itk_img.GetSize())
        if size[3] > 1:
            size[3] = 0
            extract = sitk.ExtractImageFilter()
            extract.SetSize(list(itk_img.GetSize()[0:3]) + [0])
            extract.SetIndex([0, 0, 0, 0])
            itk_img = extract.Execute(itk_img)
        else:
            itk_img = itk_img[:,:,:,0]

    # 2. Reorient to LPS — DICOM standard, corresponds directly to NIfTI LAS
    #     scanner-native (affine diagonal stays [-1,-1,+1]), no rotation artifacts.
    #     RIP creates a 90° Y/Z rotation producing non-diagonal affine matrices,
    #     which corrupts coordinate calculations (max diameter, RAS centroids).
    try:
        itk_img = sitk.DICOMOrient(itk_img, "LPS")
    except Exception as e:
        print(f"  ⚠️ Reorient fail for {in_path.name}: {e}")

    # 3. Resample to 1 mm isotropic
    original_spacing = itk_img.GetSpacing()
    original_size   = itk_img.GetSize()

    new_spacing = (1.0, 1.0, 1.0)
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetOutputOrigin(itk_img.GetOrigin())

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampled = resampler.Execute(itk_img)
    sitk.WriteImage(resampled, str(out_path))
    return out_path

def register_to_reference(ref_path, moving_path, out_path):
    """Rigid registration of moving to ref. Returns None on failure."""
    if not ref_path.exists() or not moving_path.exists():
        return None

    # Skip if already done and up-to-date
    if out_path.exists() and out_path.stat().st_size > 10_000:
        if out_path.stat().st_mtime >= max(ref_path.stat().st_mtime, moving_path.stat().st_mtime):
            return out_path

    try:
        ref    = sitk.ReadImage(str(ref_path), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

        # Initialize transform (Center-based)
        initial_transform = sitk.CenteredTransformInitializer(
            ref, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Registration framework
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=100
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Execute — SimpleITK optimizer can raise RuntimeError on divergence
        final_transform = registration_method.Execute(ref, moving)
        resampled = sitk.Resample(moving, ref, final_transform, sitk.sitkLinear, 0.0)
        sitk.WriteImage(resampled, str(out_path))
        return out_path
    except Exception as e:
        # SimpleITK optimizer failures raise exceptions (not returning None).
        # Catch them here so the caller can handle fallback gracefully.
        print(f"  ⚠️ Registration failed for {moving_path.name}: {e}")
        return None

def main():
    logger = setup_logging()
    logger.info("Starting Stage 1b: BraTS Pre-processing (LPS + Registration + Masking)")
    
    sess = get_session()
    paths = get_paths(sess)
    monai_dir = Path(paths["monai_dir"])
    
    sequences = ["t1", "t1c", "t2", "flair"]
    
    # 1. Independent Reorient and Resample (to 1mm grid)
    logger.info("Step 1: SimpleITK LPS Reorientation & 1mm Resampling")
    resampled_paths = {}
    for seq in sequences:
        raw_path = monai_dir / f"{seq}.nii.gz"
        out_path = monai_dir / f"{seq}_resampled.nii.gz"
        if raw_path.exists():
            reorient_resample(raw_path, out_path)
            resampled_paths[seq.upper()] = out_path
            logger.info(f"  Resampled {seq} -> {out_path.name}")

    # 2. Inter-modal Registration (Align T1, T2, FLAIR to T1c)
    logger.info("Step 2: Inter-modal Registration (Alignment)")
    ref_seq = "T1C" if "T1C" in resampled_paths else "T1"
    ref_path = resampled_paths[ref_seq]
    logger.info(f"  Reference sequence: {ref_seq}")

    registered_paths = {ref_seq: ref_path}
    registration_failures = []
    for seq, p in resampled_paths.items():
        if seq == ref_seq: continue
        out_path = monai_dir / f"{seq.lower()}_registered.nii.gz"
        logger.info(f"  Registering {seq} to {ref_seq}...")
        res = register_to_reference(ref_path, p, out_path)
        if res:
            registered_paths[seq] = out_path
        else:
            logger.warning(f"  Registration failed for {seq}, using independent resampling.")
            registered_paths[seq] = p
            registration_failures.append(seq)

    # Propagate registration status to session metadata for downstream stages
    if registration_failures:
        fail_str = ", ".join(registration_failures)
        logger.warning(f"  ⚠️ Alignment compromised — downstream stages notified.")
        # Write a flag file that downstream stages can check
        flag_path = monai_dir / "registration_warnings.json"
        with open(flag_path, "w") as f:
            json.dump({
                "registration_failed": True,
                "failed_sequences": registration_failures,
                "note": f"{fail_str} used non-registered resampled volumes. Volumes may be misaligned."
            }, f, indent=2)

    # 3. Multi-Contrast Brain Masking (on Aligned Data)
    logger.info("Step 3: Multi-Contrast Brain Masking on Aligned Stack")
    volumes = {}
    ref_img = None
    for seq, p in registered_paths.items():
        img = nib.load(p)
        volumes[seq] = img.get_fdata()
        if ref_img is None:
            ref_img = img
            
    # Calculate voxel volume for cc conversion
    pixdim = ref_img.header.get_zooms()
    vox_vol_cc = (pixdim[0] * pixdim[1] * pixdim[2]) / 1000.0
    
    mask = robust_brain_mask(volumes, vox_vol_cc=vox_vol_cc)
    logger.info(f"  Brain volume: {mask.sum() * vox_vol_cc:.1f} cc")
    
    # Save the mask
    mask_path = monai_dir / "brain_mask.nii.gz"
    save_nifti(mask, mask_path, ref_img)
    logger.info(f"  Saved mask: {mask_path.name}")

    # 4. Apply masking to ALL volumes & Final Cleanup
    logger.info("Step 4: Applying Skull-Stripping and Final Output")
    for seq, vol in volumes.items():
        masked = vol * mask.astype(np.float32)
        # BraTS naming standard: resampled is the FINAL preprocessed file
        out_p = monai_dir / f"{seq.lower()}_resampled.nii.gz"
        save_nifti(masked, out_p, ref_img)
        logger.info(f"  Final BraTS Volume: {out_p.name}")

    logger.info("Stage 1b completed successfully.")

if __name__ == "__main__":
    main()
