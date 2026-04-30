"""
session_loader.py
==================
Backward compatibility bridge for pybrain.io.session.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pybrain.io.session import get_session, get_paths, get_patient
except ImportError:
    # Fallback if package structure is not yet fully recognized
    from pybrain.io.session import get_session

# Backward compatibility alias
load_session = get_session


def get_seg_dir(sess: dict) -> Path:
    """Legacy helper for session-aware segmentation directory."""
    from pybrain.io.session import get_paths as _get_paths

    paths = _get_paths(sess)
    return paths.get("seg_dir", Path(sess.get("output_dir", "")))
