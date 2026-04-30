# pybrain/core/preprocessing.py
"""
MRI Preprocessing Pipeline for PY-BRAIN v2
==========================================
Handles DICOM auto-conversion, LPS reorientation, 1 mm isotropic
resampling, inter-modal registration, and brain masking.

The reference sequence is chosen dynamically: the sequence with the
most slices (best 3-D coverage) wins, with a hard preference for
T1 MPRAGE when it has ≥ 100 slices.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── DICOM detection ─────────────────────────────────────────────────────────


def _is_dicom_dir(path: Path) -> bool:
    """Return True if *path* looks like a DICOM folder."""
    if not path.is_dir():
        return False
    dcm_exts = {".dcm", ".ima", ".img", ""}
    for f in path.iterdir():
        if f.is_file() and f.suffix.lower() in dcm_exts:
            return True
    return False


def _convert_dicom_dir_to_nifti(
    dicom_dir: Path,
    out_path: Path,
    dcm2niix: str = "dcm2niix",
) -> bool:
    """Convert one DICOM series folder to a NIfTI file."""
    stem = out_path.name.replace(".nii.gz", "")
    cmd = [
        dcm2niix,
        "-o",
        str(out_path.parent),
        "-f",
        stem,
        "-z",
        "y",
        "-b",
        "y",
        "-m",
        "y",
        "-l",
        "y",  # Scale to full dynamic range (fixes intensity normalization)
        "-v",
        "0",
        str(dicom_dir),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return out_path.exists() and out_path.stat().st_size > 10_000


# ─── Geometry helpers ────────────────────────────────────────────────────────


def _count_slices(path: Path) -> int:
    """Return the number of slices (Z dimension) of a NIfTI file."""
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        return img.GetSize()[2]
    except Exception:
        return 0


def _safe_read_sitk(path: Path):
    """Read image and flatten 4-D → 3-D if needed."""
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if img.GetDimension() == 4:
        ex = sitk.ExtractImageFilter()
        sz = list(img.GetSize())
        sz[3] = 0
        ex.SetSize(sz)
        ex.SetIndex([0, 0, 0, 0])
        img = ex.Execute(img)
    return sitk.Cast(img, sitk.sitkFloat32)


# ─── Registration ─────────────────────────────────────────────────────────────


def _register_to_reference(
    ref_path: Path,
    moving_path: Path,
    out_path: Path,
) -> Optional[float]:
    """Rigid registration of *moving* → *ref*. Returns metric or None."""
    import SimpleITK as sitk

    try:
        ref = _safe_read_sitk(ref_path)
        moving = _safe_read_sitk(moving_path)

        init_tx = sitk.CenteredTransformInitializer(
            ref,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.10)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init_tx, inPlace=False)

        final_tx = reg.Execute(ref, moving)
        metric = reg.GetMetricValue()

        resampled = sitk.Resample(moving, ref, final_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        sitk.WriteImage(resampled, str(out_path))
        return float(metric)
    except Exception as exc:
        logger.warning(f"Registration failed for {moving_path.name}: {exc}")
        return None


# ─── Brain mask ───────────────────────────────────────────────────────────────


def compute_and_apply_brain_mask(
    volumes: Dict[str, np.ndarray],
    ref_img,
    output_dir: Path,
    ref_seq: str = "t1",
    ref_nifti_path: Optional[Path] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute brain mask preferentially from the reference 3-D sequence,
    then apply to all volumes.

    For mixed 2-D/3-D hospital data, using only the 3-D reference
    prevents oversized masks from thin 2-D sequences.

    Parameters
    ----------
    volumes : dict
        Sequence name → 3-D numpy array.
    ref_img : nib.Nifti1Image
        Reference NIfTI image (header / affine).
    output_dir : Path
        Where to save the mask and skull-stripped volumes.
    ref_seq : str
        Name of the reference sequence (e.g. "t1").
    ref_nifti_path : Path, optional
        Path to the reference NIfTI on disk — passed through to
        ``robust_brain_mask`` for HD-BET integration.
    """
    from pybrain.core.brainmask import robust_brain_mask
    from pybrain.io.nifti_io import save_nifti

    zooms = ref_img.header.get_zooms()[:3]
    vox_vol_cc = float(np.prod(zooms)) / 1000.0

    hdbet_work = output_dir / "hdbet_work"
    hdbet_work.mkdir(parents=True, exist_ok=True)

    # Build mask_volumes: prefer reference + T1c
    ref_upper = ref_seq.upper()
    if ref_upper in volumes:
        mask_volumes = {ref_upper: volumes[ref_upper]}
        t1c_key = next((k for k in volumes if k.upper() == "T1C"), None)
        if t1c_key and t1c_key != ref_upper:
            mask_volumes[t1c_key] = volumes[t1c_key]
        logger.info(f"Brain mask computed from: {list(mask_volumes.keys())}")
    else:
        mask_volumes = volumes
        logger.info(f"Brain mask computed from all sequences (ref {ref_seq} not found)")

    mask = robust_brain_mask(
        mask_volumes,
        vox_vol_cc=vox_vol_cc,
        ref_nifti_path=ref_nifti_path,
        work_dir=hdbet_work,
    )
    brain_vol_cc = float(mask.sum() * vox_vol_cc)
    logger.info(f"Brain mask: {brain_vol_cc:.0f} cc ({mask.sum():,} voxels)")

    # Sanity check — expected adult brain 900-1700 cc
    if brain_vol_cc < 900 or brain_vol_cc > 1700:
        logger.warning(
            f"Brain mask volume {brain_vol_cc:.0f} cc is outside expected range "
            f"(900-1700 cc). Check skull-stripping quality."
        )

    # Save mask
    mask_path = output_dir / "brain_mask.nii.gz"
    save_nifti(mask, mask_path, ref_img)

    # Apply mask and save skull-stripped volumes
    for seq, vol in volumes.items():
        stripped = vol * mask.astype(np.float32)
        out_path = output_dir / f"{seq}_resampled.nii.gz"
        save_nifti(stripped, out_path, ref_img)
        logger.info(f"  Saved: {out_path.name} (max={stripped.max():.0f})")

    return mask, brain_vol_cc


# ─── Main entry point ─────────────────────────────────────────────────────────


def preprocess_mri(
    assignments: Dict[str, str],
    output_dir: Path,
    nifti_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Full MRI preprocessing pipeline.

    1. Auto-convert DICOM folders to NIfTI (if needed).
    2. LPS reorientation + 1 mm isotropic resampling.
    3. Choose best reference sequence (most slices = best 3-D coverage).
    4. Register all other sequences into the reference space.
    5. Compute brain mask from reference only.
    6. Apply mask to all volumes.

    Parameters
    ----------
    assignments : dict
        Sequence role → DICOM folder or NIfTI path mapping.
    output_dir : Path
        Output directory for preprocessed files.
    nifti_dir : Path, optional
        Intermediate NIfTI directory (auto-created if None).

    Returns
    -------
    dict
        Sequence name → path of the final preprocessed NIfTI file.
    """
    nifti_dir = nifti_dir or (output_dir / "nifti")
    nifti_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: DICOM auto-conversion ─────────────────────────────────────
    seq_niftis: Dict[str, Path] = {}
    for role, src in assignments.items():
        src_path = Path(src)
        out_nii = nifti_dir / f"{role.lower()}.nii.gz"
        if src_path.is_dir() and _is_dicom_dir(src_path):
            logger.info(f"Converting DICOM for {role}: {src_path}")
            dcm2niix = shutil.which("dcm2niix") or "dcm2niix"
            _convert_dicom_dir_to_nifti(src_path, out_nii, dcm2niix)
        elif src_path.is_file() and src_path.suffix == ".gz":
            shutil.copy2(src_path, out_nii)
        if out_nii.exists():
            seq_niftis[role.upper()] = out_nii

    # ── Log geometry of all input sequences ───────────────────────────────
    import SimpleITK as sitk

    logger.info("Input sequence geometries:")
    for seq, src_path in assignments.items():
        src_p = Path(src_path)
        if src_p.exists() and src_p.is_file():
            try:
                img = sitk.ReadImage(str(src_p))
                size = img.GetSize()
                spacing = img.GetSpacing()
                tag = "3D" if size[2] >= 100 else "2D thin"
                logger.info(
                    f"  {seq:6s}: {size[0]}x{size[1]}x{size[2]} "
                    f"spacing={spacing[0]:.1f}x{spacing[1]:.1f}x{spacing[2]:.1f}mm "
                    f"{tag}"
                )
            except Exception:
                logger.info(f"  {seq:6s}: (could not read geometry)")

    # ── Step 2: LPS reorientation + 1 mm resampling ──────────────────────
    resampled: Dict[str, Path] = {}
    for seq, nii_path in seq_niftis.items():
        out_path = nifti_dir / f"{seq.lower()}_raw_resampled.nii.gz"
        try:
            itk_img = _safe_read_sitk(nii_path)
            try:
                itk_img = sitk.DICOMOrient(itk_img, "LPS")
            except Exception:
                pass

            original_spacing = itk_img.GetSpacing()
            original_size = itk_img.GetSize()
            new_spacing = (1.0, 1.0, 1.0)
            new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_size)
            resampler.SetOutputDirection(itk_img.GetDirection())
            resampler.SetOutputOrigin(itk_img.GetOrigin())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_itk = resampler.Execute(itk_img)
            sitk.WriteImage(resampled_itk, str(out_path))
            resampled[seq] = out_path
            logger.info(f"  Resampled {seq}: {new_size}")
        except Exception as exc:
            logger.warning(f"  Resample failed for {seq}: {exc}")

    # ── Step 3: Choose reference sequence ─────────────────────────────────
    # Prefer T1 MPRAGE (3D isotropic) because hospital T1c is often a thin
    # 2D acquisition (24-26 slices).
    if resampled:
        ref_seq = max(resampled.keys(), key=lambda s: _count_slices(resampled[s]))
        # If T1 has >= 100 slices, always prefer it (it's the MPRAGE)
        if "T1" in resampled and _count_slices(resampled["T1"]) >= 100:
            ref_seq = "T1"
    else:
        ref_seq = "T1C" if "T1C" in resampled else "T1"

    ref_path = resampled.get(ref_seq)
    if ref_path is None:
        raise RuntimeError(f"Reference sequence '{ref_seq}' not found in resampled paths")

    logger.info(f"Registration reference: {ref_seq} ({_count_slices(ref_path)} slices)")

    # ── Step 4: Register all sequences to reference ───────────────────────
    registered: Dict[str, Path] = {ref_seq: ref_path}
    registration_failures = []
    for seq, p in resampled.items():
        if seq == ref_seq:
            continue
        out_path = nifti_dir / f"{seq.lower()}_resampled.nii.gz"
        logger.info(f"  Registering {seq} to {ref_seq}...")
        metric = _register_to_reference(ref_path, p, out_path)
        if metric is not None:
            registered[seq] = out_path
            logger.info(f"    metric={metric:.4f}")
        else:
            logger.warning(f"  Registration failed for {seq}, using resampled fallback")
            registered[seq] = p
            registration_failures.append(seq)

    if registration_failures:
        flag_path = nifti_dir / "registration_warnings.json"
        with open(flag_path, "w") as f:
            json.dump(
                {
                    "registration_failed": True,
                    "failed_sequences": registration_failures,
                    "note": f"{', '.join(registration_failures)} used non-registered volumes",
                },
                f,
                indent=2,
            )

    # ── Step 5: Load volumes ──────────────────────────────────────────────
    import nibabel as nib

    volumes: Dict[str, np.ndarray] = {}
    ref_img = None
    for seq, p in registered.items():
        img = nib.load(str(p))
        volumes[seq] = img.get_fdata().astype(np.float32)
        if ref_img is None:
            ref_img = img

    # ── Step 6: Brain mask + skull strip ──────────────────────────────────
    mask, brain_vol_cc = compute_and_apply_brain_mask(
        volumes,
        ref_img,
        output_dir,
        ref_seq=ref_seq.lower(),
        ref_nifti_path=registered.get(ref_seq),
    )

    # Cleanup intermediate raw resampled files
    for seq, p in registered.items():
        if "_raw_resampled" in p.name and p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    logger.info(f"Preprocessing complete — brain {brain_vol_cc:.0f} cc")
    return registered
