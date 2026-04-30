"""
Shared pytest fixtures for PY-BRAIN test suite.

Adapted from BrainLesion/BraTS test infrastructure patterns:
- Temporary directory management with cleanup
- Synthetic NIfTI volume generation
- Mock segmentation and probability maps
- Configuration loading fixtures
"""

import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after the test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d)


@pytest.fixture
def synthetic_nifti(tmp_dir):
    """Factory fixture: create a synthetic NIfTI file with given shape and data."""

    def _create(
        filename: str = "test.nii.gz",
        shape: tuple = (240, 240, 155),
        data: np.ndarray = None,
        dtype: np.dtype = np.float32,
        spacing: tuple = (1.0, 1.0, 1.0),
    ) -> Path:
        path = tmp_dir / filename
        if data is None:
            data = np.random.rand(*shape).astype(dtype)
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(path))
        return path

    return _create


@pytest.fixture
def synthetic_brats_case(tmp_dir, synthetic_nifti):
    """Create a synthetic BraTS case with 4 modalities + optional segmentation."""
    shape = (240, 240, 155)
    subject_dir = tmp_dir / "BraTS2021_00000"
    subject_dir.mkdir(parents=True, exist_ok=True)

    modalities = {}
    for mod_name, fname in [
        ("flair", "BraTS2021_00000_flair.nii.gz"),
        ("t1", "BraTS2021_00000_t1.nii.gz"),
        ("t1c", "BraTS2021_00000_t1ce.nii.gz"),
        ("t2", "BraTS2021_00000_t2.nii.gz"),
    ]:
        data = np.random.rand(*shape).astype(np.float32) * 100
        path = synthetic_nifti(filename=fname, shape=shape, data=data)
        # Move to subject directory
        dest = subject_dir / fname
        shutil.move(str(path), str(dest))
        modalities[mod_name] = dest

    return {
        "dir": subject_dir,
        "modalities": modalities,
        "shape": shape,
    }


@pytest.fixture
def synthetic_segmentation(tmp_dir):
    """Factory: create a synthetic BraTS segmentation with labels {0,1,2,3}."""

    def _create(
        filename: str = "seg.nii.gz",
        shape: tuple = (240, 240, 155),
        has_tumor: bool = True,
    ) -> Path:
        seg = np.zeros(shape, dtype=np.int32)
        if has_tumor:
            # Create a small spherical tumor in the center
            cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
            for x in range(cx - 15, cx + 15):
                for y in range(cy - 15, cy + 15):
                    for z in range(cz - 10, cz + 10):
                        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                        if dist < 8:
                            seg[x, y, z] = 3  # ET
                        elif dist < 12:
                            seg[x, y, z] = 1  # NCR
                        elif dist < 15:
                            seg[x, y, z] = 2  # ED

        path = tmp_dir / filename
        img = nib.Nifti1Image(seg.astype(np.int32), np.eye(4))
        nib.save(img, str(path))
        return path

    return _create


@pytest.fixture
def synthetic_prob_map():
    """Factory: create a synthetic probability map (3, D, H, W)."""

    def _create(
        shape: tuple = (240, 240, 155),
        channel_means: tuple = (0.1, 0.15, 0.05),
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.RandomState(seed)
        prob = np.zeros((3,) + shape, dtype=np.float32)
        for ch, mean in enumerate(channel_means):
            prob[ch] = np.clip(rng.normal(mean, 0.1, shape), 0, 1).astype(np.float32)
        return prob

    return _create


@pytest.fixture
def mock_config():
    """Load the project's defaults.yaml config for testing."""
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    # Fallback minimal config
    return {
        "thresholds": {"wt": 0.40, "tc": 0.35, "et": 0.35},
        "ensemble_weights": {
            "segresnet": 0.60,
            "tta4": 0.40,
            "swinunetr": 0.0,
            "nnunet": 0.0,
        },
        "clinical": {"match_tolerance": 0.15, "entropy_threshold": 0.55},
    }


@pytest.fixture
def brain_mask(synthetic_brats_case):
    """Create a simple brain mask from the first modality."""
    first_mod = next(iter(synthetic_brats_case["modalities"].values()))
    data = nib.load(str(first_mod)).get_fdata()
    return (data > np.percentile(data, 10)).astype(np.float32)
