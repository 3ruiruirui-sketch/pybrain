# pybrain/io/logging_utils.py
"""
Structured logging for the PY-BRAIN pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(output_dir: Optional[Path] = None, log_level: int = logging.INFO):
    """
    Configures logging to console and optionally to a file in the session directory.
    """
    logger = logging.getLogger("pybrain")
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if output_dir is provided)
    if output_dir:
        log_file = output_dir / "session.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str):
    """Get a child logger."""
    return logging.getLogger(f"pybrain.{name}")
