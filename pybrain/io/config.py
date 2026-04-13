# pybrain/io/config.py
"""
Configuration loader for pybrain using YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def get_config() -> Dict[str, Any]:
    """Load and merge default and hardware configurations."""
    config_dir = PROJECT_ROOT / "pybrain" / "config"
    defaults = load_yaml(config_dir / "defaults.yaml")
    hardware = load_yaml(config_dir / "hardware_profiles.yaml")
    
    # Merge hardware profile
    import torch
    device_type = "cpu"
    if torch.backends.mps.is_available():
        device_type = "mps"
    elif torch.cuda.is_available():
        device_type = "cuda"
    
    # Use the hardware settings directly from hardware_profiles.yaml
    hw_settings = hardware.get("profiles", {}).get(device_type, {})
    
    config = {
        "thresholds": defaults.get("thresholds", {}),
        "ensemble_weights": defaults.get("ensemble_weights", {}),
        "clinical": defaults.get("clinical", {}),
        "ct_boost": defaults.get("ct_boost", {}),
        "labels": defaults.get("labels", {}),
        "models": defaults.get("models", {}),
        "visualizations": defaults.get("visualizations", {}),
        "hardware": {
            "device": device_type,
            "model_device": hw_settings.get("model_device", device_type),
            "mps_high_watermark": 1.4 if device_type == "mps" else None,
            "mps_low_watermark": 1.2 if device_type == "mps" else None,
            **hw_settings
        }
    }
    return config
