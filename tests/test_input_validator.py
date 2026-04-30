"""
Tests for input_validator module.

Adapted from BrainLesion/BraTS test_segmentation_algorithms.py patterns:
- Input standardization tests
- File not found handling
- Shape consistency checks
- Intensity distribution validation
"""

import shutil

import numpy as np
import pytest
from pybrain.core.input_validator import (
    ValidationResult,
    find_modality_file,
    validate_input,
    validate_input_tensor,
    validate_intensity_distribution,
    validate_nifti_loadable,
    validate_shape_consistency,
    validate_voxel_spacing,
)


class TestValidateNiftiLoadable:
    """Tests for NIfTI file loadability checks (BrainLesion/BraTS pattern)."""

    def test_valid_file(self, synthetic_nifti):
        path = synthetic_nifti(filename="valid.nii.gz", shape=(64, 64, 32))
        ok, msg = validate_nifti_loadable(path)
        assert ok
        assert "OK" in msg

    def test_missing_file(self, tmp_dir):
        path = tmp_dir / "missing.nii.gz"
        ok, msg = validate_nifti_loadable(path)
        assert not ok
        assert "not found" in msg.lower()

    def test_corrupt_file(self, tmp_dir):
        path = tmp_dir / "corrupt.nii.gz"
        path.write_bytes(b"not nifti")
        ok, msg = validate_nifti_loadable(path)
        assert not ok


class TestValidateIntensityDistribution:
    """Tests for MRI intensity distribution validation."""

    def test_normal_mri(self):
        data = np.random.rand(64, 64, 32).astype(np.float32) * 100
        ok, issues, warnings = validate_intensity_distribution(data, "t1")
        assert ok
        assert len(issues) == 0

    def test_all_zeros(self):
        data = np.zeros((64, 64, 32), dtype=np.float32)
        ok, issues, warnings = validate_intensity_distribution(data, "t1")
        assert not ok
        assert any("zero" in i for i in issues)

    def test_constant_volume(self):
        data = np.full((64, 64, 32), 50.0, dtype=np.float32)
        ok, issues, warnings = validate_intensity_distribution(data, "t1")
        assert not ok
        assert any("constant" in i for i in issues)

    def test_negative_values_warning(self):
        data = np.random.randn(64, 64, 32).astype(np.float32)
        # All negative — unusual for raw MRI
        data = data - 10
        ok, issues, warnings = validate_intensity_distribution(data, "t1")
        # Should still pass but with warnings about negative values
        assert ok
        assert any("negative" in w.lower() for w in warnings)


class TestValidateShapeConsistency:
    """Tests for cross-modality shape validation (BrainLesion/BraTS pattern)."""

    def test_consistent_shapes(self):
        volumes = {
            "t1": np.random.rand(64, 64, 32).astype(np.float32),
            "t1c": np.random.rand(64, 64, 32).astype(np.float32),
            "t2": np.random.rand(64, 64, 32).astype(np.float32),
            "flair": np.random.rand(64, 64, 32).astype(np.float32),
        }
        ok, issues = validate_shape_consistency(volumes)
        assert ok
        assert len(issues) == 0

    def test_mismatched_shapes(self):
        volumes = {
            "t1": np.random.rand(64, 64, 32).astype(np.float32),
            "t1c": np.random.rand(64, 64, 48).astype(np.float32),  # Different!
        }
        ok, issues = validate_shape_consistency(volumes)
        assert not ok
        assert any("Shape mismatch" in i for i in issues)

    def test_single_modality(self):
        volumes = {"t1": np.random.rand(64, 64, 32).astype(np.float32)}
        ok, issues = validate_shape_consistency(volumes)
        assert ok


class TestValidateVoxelSpacing:
    """Tests for voxel spacing consistency."""

    def test_consistent_spacing(self, tmp_dir, synthetic_nifti):
        paths = {}
        for mod in ["t1", "t1c"]:
            p = synthetic_nifti(
                filename=f"{mod}.nii.gz",
                shape=(64, 64, 32),
                spacing=(1.0, 1.0, 1.0),
            )
            paths[mod] = p

        ok, issues, spacings = validate_voxel_spacing(paths)
        assert ok
        assert len(issues) == 0

    def test_mismatched_spacing(self, tmp_dir, synthetic_nifti):
        paths = {}
        p1 = synthetic_nifti(filename="t1.nii.gz", shape=(64, 64, 32), spacing=(1.0, 1.0, 1.0))
        paths["t1"] = p1
        p2 = synthetic_nifti(filename="t1c.nii.gz", shape=(64, 64, 32), spacing=(2.0, 1.0, 1.0))
        paths["t1c"] = p2

        ok, issues, spacings = validate_voxel_spacing(paths)
        assert not ok
        assert len(issues) > 0


class TestFindModalityFile:
    """Tests for modality file discovery with naming conventions."""

    def test_find_by_alias(self, tmp_dir):
        # Create files with various naming conventions
        (tmp_dir / "case_t1ce.nii.gz").write_bytes(b"")
        found = find_modality_file(tmp_dir, "t1c")
        assert found is not None
        assert "t1ce" in found.name

    def test_find_flair(self, tmp_dir):
        (tmp_dir / "BraTS_flair.nii.gz").write_bytes(b"")
        found = find_modality_file(tmp_dir, "flair")
        assert found is not None

    def test_missing_modality(self, tmp_dir):
        found = find_modality_file(tmp_dir, "t1")
        assert found is None


class TestValidateInput:
    """Integration tests for full input validation (BrainLesion/BraTS input_sanity_check)."""

    def test_valid_subject(self, synthetic_brats_case):
        results = validate_input(
            synthetic_brats_case["dir"],
            check_spacing=False,
        )
        assert len(results) == 4
        for modality, result in results.items():
            assert result.passed, f"{modality} failed: {result.issues}"

    def test_missing_modality(self, synthetic_brats_case):
        # Remove one modality
        t1_path = synthetic_brats_case["modalities"]["t1"]
        t1_path.unlink()

        results = validate_input(synthetic_brats_case["dir"], check_spacing=False)
        assert not results["t1"].passed
        assert any("not found" in i for i in results["t1"].issues)
        # Others should still pass
        assert results["t1c"].passed
        assert results["t2"].passed
        assert results["flair"].passed

    def test_empty_subject_dir(self, tmp_dir):
        empty_dir = tmp_dir / "empty"
        empty_dir.mkdir()
        results = validate_input(empty_dir)
        for _modality, result in results.items():
            assert not result.passed

    def test_shape_mismatch_detected(self, tmp_dir, synthetic_nifti):
        """BrainLesion/BraTS pattern: detect shape mismatch before stacking."""
        subject_dir = tmp_dir / "subject"
        subject_dir.mkdir()
        # Different shapes for different modalities
        for mod, shape in [("t1", (64, 64, 32)), ("t1c", (64, 64, 48))]:
            p = synthetic_nifti(filename=f"{mod}.nii.gz", shape=shape)
            dest = subject_dir / f"{mod}.nii.gz"
            shutil.move(str(p), str(dest))

        results = validate_input(subject_dir, check_spacing=False)
        # Both should fail due to shape mismatch
        for mod in ["t1", "t1c"]:
            assert any(
                "Shape mismatch" in i or "mismatch" in i.lower() for i in results[mod].issues + results[mod].warnings
            )


class TestValidateInputTensor:
    """Tests for preprocessed tensor validation before model inference."""

    def test_valid_tensor(self):
        tensor = np.random.rand(1, 4, 128, 128, 128).astype(np.float32)
        ok, issues = validate_input_tensor(tensor, expected_channels=4)
        assert ok
        assert len(issues) == 0

    def test_valid_4d_tensor(self):
        tensor = np.random.rand(4, 128, 128, 128).astype(np.float32)
        ok, issues = validate_input_tensor(tensor, expected_channels=4)
        assert ok

    def test_wrong_channels(self):
        tensor = np.random.rand(1, 3, 128, 128, 128).astype(np.float32)
        ok, issues = validate_input_tensor(tensor, expected_channels=4)
        assert not ok
        assert any("channels" in i for i in issues)

    def test_nan_values(self):
        tensor = np.random.rand(1, 4, 32, 32, 32).astype(np.float32)
        tensor[0, 0, 0, 0, 0] = np.nan
        ok, issues = validate_input_tensor(tensor)
        assert not ok
        assert any("NaN" in i for i in issues)

    def test_inf_values(self):
        tensor = np.random.rand(1, 4, 32, 32, 32).astype(np.float32)
        tensor[0, 1, 0, 0, 0] = np.inf
        ok, issues = validate_input_tensor(tensor)
        assert not ok
        assert any("Inf" in i for i in issues)

    def test_zero_channel(self):
        tensor = np.random.rand(1, 4, 32, 32, 32).astype(np.float32)
        tensor[0, 2] = 0  # Zero out channel 2
        ok, issues = validate_input_tensor(tensor)
        assert not ok
        assert any("Channel 2" in i for i in issues)

    def test_shape_mismatch(self):
        tensor = np.random.rand(1, 4, 64, 64, 64).astype(np.float32)
        ok, issues = validate_input_tensor(tensor, expected_channels=4, expected_shape=(128, 128, 128))
        assert not ok
        assert any("spatial shape" in i for i in issues)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_defaults(self):
        r = ValidationResult(passed=True)
        assert r.passed
        assert r.issues == []
        assert r.warnings == []

    def test_to_dict(self):
        r = ValidationResult(
            passed=False,
            modality="t1",
            issues=["issue1"],
            warnings=["warn1"],
            stats={"shape": [64, 64, 32]},
        )
        d = r.to_dict()
        assert d["passed"] is False
        assert d["modality"] == "t1"
        assert d["stats"]["shape"] == [64, 64, 32]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
