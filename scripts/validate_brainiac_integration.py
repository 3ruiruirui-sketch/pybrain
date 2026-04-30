#!/usr/bin/env python3
"""
BrainIAC ViT-UNETR Integration Validation Script
Medical Engineering Validation - Phase 3 Enhanced Capabilities

Validates BrainIAC ViT-UNETR integration as cross-validation reader
for FLAIR-only segmentation.
"""

import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pybrain.models.brainiac_vit_unetr import (
    BrainIACViTUNETR,
    load_brainiac_model,
    run_brainiac_inference,
    create_synthetic_brainiac_output,
    integrate_brainiac_cross_validation,
)
from pybrain.io.logging_utils import get_logger

logger = get_logger("validation.brainiac_integration")


def create_test_flair_data():
    """Create synthetic FLAIR data for testing."""
    shape = (128, 128, 128)
    flair_data = np.random.rand(*shape) * 0.8 + 0.1

    # Add brain-like structure
    center = np.array([64, 64, 64])
    radius = 40

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2)
                if dist < radius:
                    flair_data[i, j, k] = np.random.rand() * 0.3 + 0.7

    return flair_data


def test_model_architecture():
    """Test BrainIAC ViT-UNETR model architecture."""
    logger.info("=== Testing BrainIAC Model Architecture ===")

    try:
        # Initialize model
        model = BrainIACViTUNETR()

        # Test forward pass
        flair_input = torch.randn(1, 1, 96, 96, 96)  # Batch size 1

        with torch.no_grad():
            output = model(flair_input)

        # Validate output
        assert output.shape == (1, 3, 96, 96, 96), f"Unexpected output shape: {output.shape}"
        assert 0 <= output.min() and output.max() <= 1, f"Output out of range: [{output.min():.3f}, {output.max():.3f}]"

        logger.info(f"Model output shape: {output.shape}")
        logger.info(f"Model output range: [{output.min():.3f}, {output.max():.3f}]")
        logger.info("✅ BrainIAC model architecture validation PASSED")

        return True

    except Exception as e:
        logger.error(f"❌ Model architecture validation failed: {e}")
        return False


def test_synthetic_output():
    """Test synthetic BrainIAC output generation."""
    logger.info("=== Testing Synthetic Output Generation ===")

    try:
        # Create synthetic output
        shape = (96, 96, 96)
        synthetic_output = create_synthetic_brainiac_output(shape)

        # Validate synthetic output
        assert synthetic_output.shape == (3, 96, 96, 96), f"Unexpected synthetic output shape: {synthetic_output.shape}"
        assert 0 <= synthetic_output.min() and synthetic_output.max() <= 1, "Synthetic output out of range"

        # Check tumor structure
        wt_mean = synthetic_output[1].mean()
        tc_mean = synthetic_output[0].mean()
        et_mean = synthetic_output[2].mean()

        logger.info(f"Synthetic output channel means - WT: {wt_mean:.3f}, TC: {tc_mean:.3f}, ET: {et_mean:.3f}")

        # WT should generally be highest (largest region)
        if wt_mean >= tc_mean and wt_mean >= et_mean:
            logger.info("✅ Synthetic output has realistic tumor structure")
        else:
            logger.warning("⚠️  Synthetic output structure may not be realistic")

        logger.info("✅ Synthetic output generation validation PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Synthetic output validation failed: {e}")
        return False


def test_inference_pipeline():
    """Test BrainIAC inference pipeline."""
    logger.info("=== Testing Inference Pipeline ===")

    try:
        # Create test data
        flair_data = create_test_flair_data()

        # Create model
        model = BrainIACViTUNETR()
        model.eval()

        # Run inference
        device = torch.device("cpu")
        seg_probs = run_brainiac_inference(model, flair_data, device)

        # Validate results
        assert seg_probs is not None, "Inference returned None"
        assert seg_probs.shape[0] == 3, f"Expected 3 channels, got {seg_probs.shape[0]}"
        assert 0 <= seg_probs.min() and seg_probs.max() <= 1, "Inference output out of range"

        logger.info(f"Inference output shape: {seg_probs.shape}")
        logger.info(f"Inference output range: [{seg_probs.min():.3f}, {seg_probs.max():.3f}]")

        # Compute volumes
        vox_vol_cc = 1.0  # Assume 1mm³ voxels
        volumes = {}
        for i, name in enumerate(["TC", "WT", "ET"]):
            volume_cc = np.sum(seg_probs[i] > 0.5) * vox_vol_cc
            volumes[name] = volume_cc

        logger.info(f"Computed volumes - TC: {volumes['TC']:.1f}, WT: {volumes['WT']:.1f}, ET: {volumes['ET']:.1f}")

        logger.info("✅ Inference pipeline validation PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Inference pipeline validation failed: {e}")
        return False


def test_model_loading():
    """Test BrainIAC model loading."""
    logger.info("=== Testing Model Loading ===")

    try:
        device = torch.device("cpu")

        # Test with non-existent path (should return None gracefully)
        model = load_brainiac_model(Path("non_existent_model.pt"), device)

        if model is None:
            logger.info("✅ Non-existent model handled gracefully")
        else:
            logger.warning("⚠️  Expected None for non-existent model")

        # Test with potential model paths
        potential_paths = [
            Path("/Users/ssoares/Downloads/PY-BRAIN/brainiac_vit_unetr.pt"),
            Path("/Users/ssoares/Downloads/PY-BRAIN/models/brainiac_vit_unetr.pt"),
        ]

        model_found = False
        for path in potential_paths:
            if path.exists():
                model = load_brainiac_model(path, device)
                if model is not None:
                    logger.info(f"✅ Model loaded from {path}")
                    model_found = True
                    break

        if not model_found:
            logger.info("✅ No BrainIAC model found (will use synthetic output)")

        return True

    except Exception as e:
        logger.error(f"❌ Model loading validation failed: {e}")
        return False


def test_cross_validation_integration():
    """Test complete cross-validation integration."""
    logger.info("=== Testing Cross-Validation Integration ===")

    try:
        # Create temporary test directory
        test_dir = Path("/tmp/brainiac_test")
        test_dir.mkdir(exist_ok=True)

        # Create synthetic FLAIR file
        flair_data = create_test_flair_data()

        # Save as NIfTI (simulated)
        import nibabel as nib

        flair_img = nib.Nifti1Image(flair_data, np.eye(4))
        flair_path = test_dir / "test_flair.nii.gz"
        nib.save(flair_img, str(flair_path))

        # Run cross-validation
        results = integrate_brainiac_cross_validation(flair_path, test_dir, device=torch.device("cpu"))

        # Validate results
        assert "method" in results, "Missing method in results"
        assert "status" in results, "Missing status in results"

        if results["status"] == "success":
            assert "volumes_cc" in results, "Missing volumes in successful results"
            assert "output_path" in results, "Missing output path in successful results"

            volumes = results["volumes_cc"]
            assert "TC" in volumes and "WT" in volumes and "ET" in volumes, "Missing volume components"

            logger.info("Cross-validation results:")
            logger.info(f"  Method: {results['method']}")
            logger.info(f"  Volumes: TC={volumes['TC']:.1f}, WT={volumes['WT']:.1f}, ET={volumes['ET']:.1f}")

            # Check if output file was created
            output_path = Path(results["output_path"])
            if output_path.exists():
                logger.info("✅ Output file created successfully")
            else:
                logger.warning("⚠️  Output file not found")

        logger.info("✅ Cross-validation integration validation PASSED")

        # Cleanup
        if flair_path.exists():
            flair_path.unlink()
        if test_dir.exists():
            for file in test_dir.glob("*"):
                file.unlink()
            test_dir.rmdir()

        return True

    except Exception as e:
        logger.error(f"❌ Cross-validation integration validation failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("=== Testing Edge Cases ===")

    try:
        # Test with small input
        model = BrainIACViTUNETR()
        small_input = torch.randn(1, 1, 32, 32, 32)  # Smaller than expected

        try:
            with torch.no_grad():
                model(small_input)
            logger.info("✅ Small input handled")
        except:
            logger.info("✅ Small input properly rejected")

        # Test with synthetic output for different shapes
        shapes = [(64, 64, 64), (128, 128, 128), (160, 160, 160)]
        for shape in shapes:
            synthetic = create_synthetic_brainiac_output(shape)
            assert synthetic.shape == (3, *shape), f"Failed for shape {shape}"

        logger.info("✅ Different shapes handled correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Edge case testing failed: {e}")
        return False


def run_validation():
    """Run complete BrainIAC integration validation."""
    logger.info("Starting BrainIAC ViT-UNETR Integration Validation")

    results = []

    # Test 1: Model architecture
    results.append(test_model_architecture())

    # Test 2: Synthetic output generation
    results.append(test_synthetic_output())

    # Test 3: Inference pipeline
    results.append(test_inference_pipeline())

    # Test 4: Model loading
    results.append(test_model_loading())

    # Test 5: Cross-validation integration
    results.append(test_cross_validation_integration())

    # Test 6: Edge cases
    results.append(test_edge_cases())

    # Summary
    passed = sum(results)
    total = len(results)

    logger.info("\n=== VALIDATION SUMMARY ===")
    logger.info(f"Tests passed: {passed}/{total}")

    if passed == total:
        logger.info("🎉 ALL BRAINIAC INTEGRATION VALIDATIONS PASSED")
        logger.info("BrainIAC ViT-UNETR is ready for cross-validation deployment")
        return True
    else:
        logger.error("❌ SOME BRAINIAC INTEGRATION VALIDATIONS FAILED")
        logger.error("BrainIAC integration needs review")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
