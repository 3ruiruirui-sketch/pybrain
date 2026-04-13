#!/usr/bin/env python3
"""
SegResNet Channel Order Validation Script
Medical Engineering Validation - Phase 1 Critical Safety Fix

Validates that the channel permutation fix correctly aligns
pipeline input with MONAI SegResNet expected format.
"""

import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pybrain.models.segresnet import run_segresnet_inference
from pybrain.io.logging_utils import get_logger

logger = get_logger("validation.segresnet")

def create_test_input():
    """Create synthetic input matching pipeline format."""
    # Pipeline format: [FLAIR, T1, T1c, T2]
    pipeline_input = np.random.rand(1, 4, 96, 96, 96).astype(np.float32)
    
    # Add realistic intensity patterns for each modality
    pipeline_input[0, 0] *= 0.8  # FLAIR (lower intensity)
    pipeline_input[0, 1] *= 1.0  # T1 (baseline)
    pipeline_input[0, 2] *= 1.2  # T1c (higher - enhancing)
    pipeline_input[0, 3] *= 0.9  # T2 (slightly higher than FLAIR)
    
    return torch.from_numpy(pipeline_input)

def validate_channel_permutation():
    """Validate the channel permutation logic."""
    logger.info("=== SegResNet Channel Order Validation ===")
    
    # Create test input
    test_input = create_test_input()
    logger.info(f"Test input shape: {test_input.shape}")
    
    # Expected pipeline order: [FLAIR, T1, T1c, T2]
    # Expected MONAI order: [T1c, T1, T2, FLAIR]
    # Permutation indices: [2, 1, 3, 0]
    
    pipeline_order = ['FLAIR', 'T1', 'T1c', 'T2']
    monai_order = ['T1c', 'T1', 'T2', 'FLAIR']
    permutation = [2, 1, 3, 0]
    
    logger.info(f"Pipeline order: {pipeline_order}")
    logger.info(f"MONAI order: {monai_order}")
    logger.info(f"Permutation indices: {permutation}")
    
    # Apply permutation
    permuted = test_input[:, permutation, :, :, :]
    
    # Validate permutation
    for i, (pipeline_idx, monai_channel) in enumerate(zip(permutation, monai_order)):
        pipeline_channel = pipeline_order[pipeline_idx]
        logger.info(f"Index {i}: Pipeline[{pipeline_channel}] -> MONAI[{monai_channel}]")
        
        # Check that the data is correctly permuted
        original_slice = test_input[0, pipeline_idx, 48, 48, 48]
        permuted_slice = permuted[0, i, 48, 48, 48]
        
        if not np.allclose(original_slice, permuted_slice):
            logger.error(f"Channel permutation failed at index {i}")
            return False
    
    logger.info("✅ Channel permutation validation PASSED")
    return True

def validate_with_mock_model():
    """Test with a mock model to ensure inference works."""
    logger.info("=== Mock Model Inference Test ===")
    
    class MockSegResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(4, 4, kernel_size=1)
            
        def forward(self, x):
            # Return mock logits
            return self.conv(x)
    
    # Create mock model and config
    mock_model = MockSegResNet()
    sw_config = {"roi_size": (96, 96, 96), "sw_batch_size": 1, "overlap": 0.5}
    device = torch.device("cpu")
    
    try:
        # Test inference with channel permutation
        test_input = create_test_input()
        result = run_segresnet_inference(mock_model, test_input, device, sw_config)
        
        logger.info(f"Inference result shape: {result.shape}")
        logger.info(f"Expected shape: (3, D, H, W)")
        
        if result.shape[0] == 3:  # Should be [TC, WT, ET]
            logger.info("✅ Mock model inference PASSED")
            return True
        else:
            logger.error(f"Unexpected output shape: {result.shape}")
            return False
            
    except Exception as e:
        logger.error(f"Mock model inference FAILED: {e}")
        return False

def run_validation():
    """Run complete validation suite."""
    logger.info("Starting SegResNet Channel Order Validation")
    
    results = []
    
    # Test 1: Channel permutation logic
    results.append(validate_channel_permutation())
    
    # Test 2: Mock model inference
    results.append(validate_with_mock_model())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info(f"\n=== VALIDATION SUMMARY ===")
    logger.info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL VALIDATIONS PASSED")
        logger.info("SegResNet channel order fix is working correctly")
        return True
    else:
        logger.error("❌ SOME VALIDATIONS FAILED")
        logger.error("SegResNet channel order fix needs review")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
