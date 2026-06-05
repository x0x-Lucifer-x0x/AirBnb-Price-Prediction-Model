"""
Utils
=====
Shared utilities: logging setup, artifact persistence, config.
"""

import logging
import pickle
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("airbnb_pipeline")


def save_artifact(obj, output_dir: str, filename: str):
    path = Path(output_dir) / filename
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logging.getLogger("airbnb_pipeline").info(f"  Artifact saved → {path}")


def load_artifact(output_dir: str, filename: str):
    path = Path(output_dir) / filename
    with open(path, "rb") as f:
        return pickle.load(f)