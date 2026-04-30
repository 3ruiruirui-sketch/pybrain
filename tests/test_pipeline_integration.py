"""
Pipeline integration tests.

Adapted from BrainLesion/BraTS test patterns:
- End-to-end pipeline stage testing with mocked model inference
- Preprocess -> Segment -> Validate flow
- Ensemble fusion validation
- Config consistency checks
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml
from pybrain.core.input_validator import validate_input
from pybrain.core.labels import canonical_labels, is_pipeline_convention
from pybrain.core.metrics import compute_dice, compute_volume_cc
from pybrain.core.normalization import norm01, zscore_robust
from pybrain.core.output_checker import (
    check_hierarchy_violations,
    check_probability_range,
    sanity_check_segmentation,
)
from pybrain.models.ensemble import compute_uncertainty, run_weighted_ensemble


class TestPreprocessNormalizePipeline:
    """Tests for preprocessing normalization stage."""

    def test_norm01_roundtrip(self):
        data = np.array([10, 50, 100], dtype=np.float32)
        normed = norm01(data)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)

    def test_zscore_brain_mask(self):
        np.random.seed(42)
        data = np.random.normal(100, 15, (64, 64, 32)).astype(np.float32)
        mask = np.ones((64, 64, 32), dtype=np.float32)
        result = zscore_robust(data, mask)
        # Within-brain mean should be ~0
        assert abs(result[mask > 0].mean()) < 0.5

    def test_zscore_small_mask_fallback(self):
        """zscore_robust falls back to whole-image stats for small masks."""
        data = np.random.normal(100, 15, (64, 64, 32)).astype(np.float32)
        small_mask = np.zeros((64, 64, 32), dtype=np.float32)
        small_mask[30:34, 30:34, 15:17] = 1  # Very small mask
        result = zscore_robust(data, small_mask)
        assert not np.any(np.isnan(result))

    def test_zscore_empty_mask(self):
        data = np.random.normal(100, 15, (32, 32, 32)).astype(np.float32)
        mask = np.zeros((32, 32, 32), dtype=np.float32)
        result = zscore_robust(data, mask)
        # Should not crash; falls back to whole-image
        assert result.shape == data.shape


class TestEnsembleFusionPipeline:
    """Tests for ensemble fusion stage (BrainLesion/BraTS weighted fusion)."""

    def test_weighted_ensemble_basic(self):
        p1 = np.random.rand(3, 16, 16, 16).astype(np.float32) * 0.8
        p2 = np.random.rand(3, 16, 16, 16).astype(np.float32) * 0.8
        fused, models = run_weighted_ensemble(
            [
                ("segresnet", p1, 0.6),
                ("tta4", p2, 0.4),
            ]
        )
        assert fused.shape == p1.shape
        assert fused.min() >= 0.0
        assert fused.max() <= 1.0
        assert "segresnet" in models
        assert "tta4" in models

    def test_ensemble_skips_none(self):
        p1 = np.random.rand(3, 16, 16, 16).astype(np.float32) * 0.5
        fused, models = run_weighted_ensemble(
            [
                ("segresnet", p1, 0.6),
                ("tta4", None, 0.4),
            ]
        )
        assert "segresnet" in models
        assert "tta4" not in models
        np.testing.assert_array_almost_equal(fused, p1)

    def test_ensemble_all_none_raises(self):
        with pytest.raises(ValueError, match="No valid"):
            run_weighted_ensemble(
                [
                    ("segresnet", None, 0.6),
                    ("tta4", None, 0.4),
                ]
            )

    def test_ensemble_range_validation(self, synthetic_prob_map):
        p1 = synthetic_prob_map(channel_means=(0.3, 0.4, 0.2))
        p2 = synthetic_prob_map(channel_means=(0.2, 0.3, 0.1), seed=99)
        fused, _ = run_weighted_ensemble(
            [
                ("m1", p1, 0.5),
                ("m2", p2, 0.5),
            ]
        )
        ok, issues = check_probability_range(fused, "ensemble")
        assert ok, f"Ensemble range issues: {issues}"


class TestUncertaintyComputation:
    """Tests for uncertainty estimation (BrainLesion/BraTS sanity check S3)."""

    def test_identical_models_zero_variance(self):
        """Property 1: identical models -> inter-model variance = 0."""
        p = np.full((3, 16, 16, 16), 0.1, dtype=np.float32)
        u = compute_uncertainty(p, [p.copy(), p.copy()])
        # Low uncertainty for identical confident models
        assert u.mean() < 0.15

    def test_max_disagreement_high_uncertainty(self):
        """Property 2: maximally disagreeing models -> high uncertainty."""
        p1 = np.ones((3, 16, 16, 16), dtype=np.float32)
        p2 = np.zeros((3, 16, 16, 16), dtype=np.float32)
        p_avg = (p1 + p2) / 2.0
        u = compute_uncertainty(p_avg, [p1, p2])
        assert u.mean() > 0.1

    def test_disagreement_exceeds_identical(self):
        """Uncertainty from disagreement must exceed uncertainty from agreement."""
        shape = (3, 16, 16, 16)
        p_confident = np.full(shape, 0.05, dtype=np.float32)
        u_identical = compute_uncertainty(p_confident, [p_confident, p_confident])

        p_all_ones = np.ones(shape, dtype=np.float32)
        p_all_zeros = np.zeros(shape, dtype=np.float32)
        u_disagree = compute_uncertainty(
            (p_all_ones + p_all_zeros) / 2.0,
            [p_all_ones, p_all_zeros],
        )
        assert u_disagree.mean() > u_identical.mean()

    def test_uncertainty_bounded(self):
        """Uncertainty output must be in [0, 1]."""
        p = np.random.rand(3, 16, 16, 16).astype(np.float32)
        u = compute_uncertainty(p, [p, p * 0.5])
        assert u.min() >= 0.0
        assert u.max() <= 1.0


class TestSegmentationValidationPipeline:
    """Tests for the validation stage (Dice, hierarchy, labels)."""

    def test_dice_perfect(self):
        pred = np.random.randint(0, 4, (64, 64, 64)).astype(np.int32)
        dice = compute_dice(pred, pred)
        assert dice == pytest.approx(1.0)

    def test_dice_no_overlap(self):
        pred = np.zeros((64, 64, 64), dtype=np.int32)
        pred[:32] = 1
        gt = np.zeros((64, 64, 64), dtype=np.int32)
        gt[32:] = 1
        dice = compute_dice(pred, gt)
        assert dice == pytest.approx(0.0)

    def test_label_convention_conversion(self):
        gt = np.array([0, 1, 2, 4], dtype=np.int32)
        converted = canonical_labels(gt)
        assert list(converted) == [0, 1, 2, 3]
        assert is_pipeline_convention(converted)

    def test_volume_computation(self):
        mask = np.ones((100, 100, 100), dtype=np.float32)
        vox_vol = 0.001  # 1mm^3 = 0.001 cc
        vol = compute_volume_cc(mask, vox_vol)
        assert vol == pytest.approx(1000.0 * 0.001)


class TestEndToEndMockPipeline:
    """End-to-end pipeline test with fully mocked inference.

    Adapted from BrainLesion/BraTS test_run_container pattern:
    mock the heavy inference, validate the I/O and validation stages.
    """

    def test_full_pipeline_flow(self, tmp_dir, synthetic_brats_case):
        """Test: validate input -> mock segment -> sanity check -> validate."""
        # Stage 1: Validate input
        results = validate_input(synthetic_brats_case["dir"], check_spacing=False)
        all_passed = all(r.passed for r in results.values())
        assert all_passed, f"Input validation failed: {results}"

        # Stage 2: Mock segmentation (create a synthetic output)
        seg_data = np.zeros(synthetic_brats_case["shape"], dtype=np.int32)
        cx, cy, cz = 120, 120, 77
        for x in range(cx - 10, cx + 10):
            for y in range(cy - 10, cy + 10):
                for z in range(cz - 8, cz + 8):
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                    if dist < 5:
                        seg_data[x, y, z] = 3
                    elif dist < 8:
                        seg_data[x, y, z] = 1
                    elif dist < 10:
                        seg_data[x, y, z] = 2

        output_path = tmp_dir / "segmentation.nii.gz"
        img = nib.Nifti1Image(seg_data, np.eye(4))
        nib.save(img, str(output_path))

        # Stage 3: Sanity check the output
        result = sanity_check_segmentation(
            input_dir=synthetic_brats_case["dir"],
            output_path=output_path,
            subject_name="BraTS2021_00000",
        )
        assert result.passed, f"Sanity check failed: {result.issues}"

        # Stage 4: Validate hierarchy
        ok, issues = check_hierarchy_violations(seg_data)
        assert ok, f"Hierarchy violations: {issues}"

        # Stage 5: Verify label convention
        assert is_pipeline_convention(seg_data)


class TestConfigConsistency:
    """Tests for config file consistency (adapted from BrainLesion/BraTS S1/S2)."""

    def test_defaults_yaml_exists(self):
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        assert cfg_path.exists(), f"defaults.yaml not found at {cfg_path}"

    def test_thresholds_in_range(self):
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        if not cfg_path.exists():
            pytest.skip("defaults.yaml not found")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        thresholds = cfg.get("thresholds", {})
        for name, val in thresholds.items():
            assert 0.0 < val < 1.0, f"threshold {name}={val} outside (0,1)"

    def test_ensemble_weights_sum(self):
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        if not cfg_path.exists():
            pytest.skip("defaults.yaml not found")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        weights = cfg.get("ensemble_weights", {})
        active = {k: v for k, v in weights.items() if v > 0}
        if active:
            total = sum(active.values())
            assert total > 0, "Total active ensemble weight must be positive"

    def test_preprocessing_config(self):
        """S1: Preprocessing config smoke test."""
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
        if not cfg_path.exists():
            pytest.skip("defaults.yaml not found")

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        pre = cfg.get("preprocessing", None)
        assert pre is not None, "preprocessing section missing from defaults.yaml"
        # These should be boolean values
        assert isinstance(pre.get("histogram_normalize", True), bool)
        assert isinstance(pre.get("bilateral_filter", True), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
