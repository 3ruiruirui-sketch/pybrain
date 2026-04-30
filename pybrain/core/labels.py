"""BraTS label convention utilities.

Pipeline convention: 0=background, 1=necrotic, 2=edema, 3=enhancing
BraTS2021 GT:        0=background, 1=necrotic, 2=edema, 4=enhancing
"""

import numpy as np


def canonical_labels(seg: np.ndarray) -> np.ndarray:
    """Remap BraTS2021 GT labels {0,1,2,4} → pipeline convention {0,1,2,3}.

    Args:
        seg: Segmentation array with BraTS2021 or pipeline labels

    Returns:
        Segmentation with pipeline convention (ET=3)
    """
    out = seg.astype(np.int32).copy()
    out[out == 4] = 3
    return out


def is_pipeline_convention(seg: np.ndarray) -> bool:
    """Check if segmentation uses pipeline label convention.

    Args:
        seg: Segmentation array

    Returns:
        True if all labels are in {0, 1, 2, 3}
    """
    unique = np.unique(seg.astype(np.int32))
    return all(label in [0, 1, 2, 3] for label in unique)


def is_brats_convention(seg: np.ndarray) -> bool:
    """Check if segmentation uses BraTS2021 label convention.

    Args:
        seg: Segmentation array

    Returns:
        True if label 4 exists and label 3 does not
    """
    unique = np.unique(seg.astype(np.int32))
    return 4 in unique and 3 not in unique


def get_label_names() -> dict:
    """Get human-readable label names for pipeline convention.

    Returns:
        Dictionary mapping label IDs to names
    """
    return {
        0: "Background",
        1: "Necrotic Core (NCR)",
        2: "Peritumoral Edema (ED)",
        3: "Enhancing Tumor (ET)",
    }
