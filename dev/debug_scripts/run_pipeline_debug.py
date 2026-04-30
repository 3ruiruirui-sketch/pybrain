#!/usr/bin/env python3
"""
Debug runner for PyBrain pipeline — v1.0
========================================
Checks hardware, validates configuration, and runs a quick test of each stage
to detect CPU vs Metal usage and identify potential errors.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_debug.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def check_hardware():
    """Check available hardware (CPU, MPS/Metal, CUDA)."""
    logger.info("=" * 60)
    logger.info("🔍 HARDWARE CHECK")
    logger.info("=" * 60)

    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")

    # Check CPU
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"CPU cores: {cpu_count}")

    # Check MPS (Metal)
    if hasattr(torch.backends, "mps"):
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        logger.info(f"MPS (Metal) available: {mps_available}")
        logger.info(f"MPS (Metal) built: {mps_built}")
    else:
        mps_available = False
        mps_built = False
        logger.info("MPS (Metal) not supported in this PyTorch build")

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA available: True (devices: {torch.cuda.device_count()})")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA available: False")

    # Recommend device
    if mps_available and mps_built:
        recommended = "mps (Metal)"
    elif cuda_available:
        recommended = "cuda"
    else:
        recommended = "cpu"
    logger.info(f"✅ Recommended device: {recommended}")

    return {
        "cpu_cores": cpu_count,
        "mps_available": mps_available,
        "mps_built": mps_built,
        "cuda_available": cuda_available,
        "recommended_device": recommended,
    }


def check_config():
    """Check pybrain hardware configuration."""
    logger.info("=" * 60)
    logger.info("🔍 CONFIGURATION CHECK")
    logger.info("=" * 60)

    config_path = PROJECT_ROOT / "pybrain" / "config" / "hardware_profiles.yaml"
    if config_path.exists():
        logger.info(f"Config file: {config_path}")
        with open(config_path) as f:
            content = f.read()
            logger.info(f"Config content:\n{content}")

        # Parse YAML manually (simple)
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        device_cfg = config.get("hardware", {}).get("device", "unknown")
        model_device_cfg = config.get("hardware", {}).get("model_device", "unknown")
        logger.info(f"Config device: {device_cfg}")
        logger.info(f"Config model_device: {model_device_cfg}")
        return config
    else:
        logger.warning(f"Config file not found: {config_path}")
        return None


def check_models():
    """Check available models."""
    logger.info("=" * 60)
    logger.info("🔍 MODEL CHECK")
    logger.info("=" * 60)

    bundle_dir = PROJECT_ROOT / "models" / "brats_bundle"
    if bundle_dir.exists():
        logger.info(f"Bundle dir: {bundle_dir}")
        models = list(bundle_dir.rglob("*.pth"))
        models += list(bundle_dir.rglob("*.pt"))
        for m in models:
            size_mb = m.stat().st_size / (1024 * 1024)
            logger.info(f"  {m.name}: {size_mb:.1f} MB")
    else:
        logger.warning(f"Bundle dir not found: {bundle_dir}")


def check_scripts():
    """Check all pipeline scripts exist."""
    logger.info("=" * 60)
    logger.info("🔍 SCRIPT CHECK")
    logger.info("=" * 60)

    scripts = [
        "0_preflight_check.py",
        "0_clinical_validator.py",
        "1_dicom_to_nifti.py",
        "1b_brats_preproc.py",
        "2_ct_integration.py",
        "3_brain_tumor_analysis.py",
        "5_validate_segmentation.py",
        "6_tumour_location.py",
        "7_tumour_morphology.py",
        "8_radiomics_analysis.py",
        "8b_brainiac_prediction.py",
        "9_generate_report.py",
        "9_generate_report_pt.py",
        "10_enhanced_visualisation.py",
        "11_mricrogl_visualisation.py",
        "12_brats_figure1.py",
    ]

    scripts_dir = PROJECT_ROOT / "scripts"
    missing = []
    for s in scripts:
        path = scripts_dir / s
        if path.exists():
            logger.info(f"  ✅ {s}")
        else:
            logger.warning(f"  ❌ {s} — MISSING")
            missing.append(s)

    if missing:
        logger.warning(f"Missing scripts: {missing}")
    else:
        logger.info("All scripts present.")
    return missing


def check_dependencies():
    """Check Python dependencies."""
    logger.info("=" * 60)
    logger.info("🔍 DEPENDENCY CHECK")
    logger.info("=" * 60)

    required = [
        "torch",
        "numpy",
        "nibabel",
        "scipy",
        "matplotlib",
        "SimpleITK",
        "monai",
        "radiomics",
        "xgboost",
        "yaml",
        "tqdm",
        "requests",
    ]

    missing = []
    for pkg in required:
        try:
            if pkg == "yaml":
                import yaml
            elif pkg == "radiomics":
                import radiomics
            else:
                __import__(pkg)
            logger.info(f"  ✅ {pkg}")
        except ImportError:
            logger.warning(f"  ❌ {pkg} — NOT INSTALLED")
            missing.append(pkg)

    if missing:
        logger.warning(f"Missing packages: {missing}")
    else:
        logger.info("All dependencies present.")
    return missing


def test_device_inference(device_str: str):
    """Test inference on a specific device."""
    logger.info(f"  Testing {device_str}...")
    try:
        device = torch.device(device_str)
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        conv = torch.nn.Conv3d(1, 1, 3, padding=1).to(device)
        y = conv(x)
        logger.info(f"    ✅ {device_str}: Conv3d OK (output shape: {y.shape})")
        return True
    except Exception as e:
        logger.warning(f"    ❌ {device_str}: {e}")
        return False


def run_full_debug():
    """Run all debug checks."""
    logger.info("=" * 60)
    logger.info("🧠 PyBrain Pipeline Debug Runner")
    logger.info(f"   Started: {datetime.now().isoformat()}")
    logger.info(f"   Project: {PROJECT_ROOT}")
    logger.info("=" * 60)

    results = {}

    # 1. Hardware
    results["hardware"] = check_hardware()

    # 2. Configuration
    results["config"] = check_config()

    # 3. Models
    check_models()

    # 4. Scripts
    results["missing_scripts"] = check_scripts()

    # 5. Dependencies
    results["missing_deps"] = check_dependencies()

    # 6. Device tests
    logger.info("=" * 60)
    logger.info("🔍 DEVICE INFERENCE TEST")
    logger.info("=" * 60)
    results["device_tests"] = {}
    results["device_tests"]["cpu"] = test_device_inference("cpu")
    if results["hardware"]["mps_available"]:
        results["device_tests"]["mps"] = test_device_inference("mps")
    if results["hardware"]["cuda_available"]:
        results["device_tests"]["cuda"] = test_device_inference("cuda")

    # Summary
    logger.info("=" * 60)
    logger.info("📊 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Hardware: {results['hardware']['recommended_device']}")
    logger.info(f"Missing scripts: {len(results['missing_scripts'])}")
    logger.info(f"Missing deps: {len(results['missing_deps'])}")
    for dev, ok in results["device_tests"].items():
        status = "✅" if ok else "❌"
        logger.info(f"  {status} {dev}")

    # Save results
    output_path = PROJECT_ROOT / "pipeline_debug_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {output_path}")

    # Device recommendation
    logger.info("=" * 60)
    logger.info("🎯 DEVICE RECOMMENDATION PER STAGE")
    logger.info("=" * 60)
    stages = {
        "Stage 1 — DICOM→NIfTI": "cpu (I/O bound)",
        "Stage 1b — BraTS Refinement": "cpu (SimpleITK)",
        "Stage 2 — CT Integration": "cpu (registration)",
        "Stage 3 — AI Segmentation": f"{results['hardware']['recommended_device']} (inference)",
        "Stage 5 — Validation": "cpu (metrics)",
        "Stage 6 — Location": "cpu (analysis)",
        "Stage 7 — Morphology": "cpu (analysis)",
        "Stage 8 — Radiomics": "cpu (pyradiomics)",
        "Stage 8b — BrainIAC": f"{results['hardware']['recommended_device']} (inference)",
        "Stage 9/9b — Reports": "cpu (PDF generation)",
        "Stage 10 — Visualisation": "cpu (matplotlib)",
        "Stage 11 — MRIcroGL": "cpu (rendering)",
        "Stage 12 — BraTS Figure": "cpu (matplotlib)",
    }
    for stage, dev in stages.items():
        logger.info(f"  {stage}: {dev}")

    return results


if __name__ == "__main__":
    try:
        results = run_full_debug()
        if results["missing_scripts"] or results["missing_deps"]:
            logger.warning("\n⚠️  Some issues detected. Review the log above.")
            sys.exit(1)
        else:
            logger.info("\n✅ All checks passed.")
            sys.exit(0)
    except Exception as e:
        logger.critical(f"Debug runner failed: {e}", exc_info=True)
        sys.exit(1)