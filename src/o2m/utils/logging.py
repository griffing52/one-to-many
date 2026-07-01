"""Minimal logging helper shared by scripts and modules."""
from __future__ import annotations

import logging

_FMT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"


def get_logger(name: str = "o2m", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
