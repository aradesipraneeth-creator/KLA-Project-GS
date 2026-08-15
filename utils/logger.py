"""
Logging utilities for training tracking and CSV output.
"""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class CSVLogger:
    """Logs training metrics to a CSV file."""

    def __init__(self, filepath: Path, fieldnames: List[str]):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self.file_exists = self.filepath.exists()

        if not self.file_exists:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
