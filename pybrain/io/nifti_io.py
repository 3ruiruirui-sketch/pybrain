# pybrain/io/nifti_io.py
"""
NIfTI read/write helpers with metadata preservation.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.Nifti1Image]:
    """Loads a NIfTI file and returns the data array and the nibabel image object."""
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img


def save_nifti(data: np.ndarray, path: Path, reference_img: nib.Nifti1Image):
    """Saves data as a NIfTI file using the affine and header from a reference image."""
    out_img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(out_img, str(path))
