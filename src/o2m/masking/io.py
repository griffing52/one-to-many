"""Write masks in the COLMAP / Nerfstudio convention.

COLMAP ignores image pixels where the mask is **black (0)** and uses **white
(255)**. Our maskers return True for *foreground to exclude*, so we write the
inverse: foreground -> 0, background -> 255.

COLMAP expects the mask filename to be ``<image_filename>.png`` (the full image
name, including its extension, plus ``.png``). Nerfstudio's ``mask_path`` can
point at the same files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def colmap_mask_name(image_name: str) -> str:
    return f"{image_name}.png"


def foreground_to_colmap_mask(foreground: np.ndarray) -> np.ndarray:
    """bool foreground -> uint8 COLMAP mask (255 keep, 0 ignore)."""
    keep = ~foreground.astype(bool)
    return (keep.astype(np.uint8) * 255)


def write_colmap_masks(foreground_masks: List[np.ndarray],
                       image_names: List[str],
                       out_dir: Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fg, name in zip(foreground_masks, image_names):
        mask = foreground_to_colmap_mask(fg)
        p = out_dir / colmap_mask_name(name)
        cv2.imwrite(str(p), mask)
        written.append(p)
    return written
