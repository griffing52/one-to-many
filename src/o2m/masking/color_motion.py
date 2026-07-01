"""Model-free temporal-motion masker.

Bootstrap masker for the first end-to-end run: pixels that deviate strongly
from the per-pixel temporal median (the static background estimate) are flagged
as moving foreground. Cheap, no learned weights, good enough to keep COLMAP from
matching on the arm. Swap in a learned masker (SAM/Grounded-SAM) for quality.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np


class ColorMotionMasker:
    def __init__(self, threshold: float = 25.0, dilate_px: int = 9,
                 min_area_frac: float = 0.0):
        self.threshold = threshold
        self.dilate_px = dilate_px
        self.min_area_frac = min_area_frac
        self._background: np.ndarray | None = None

    def fit_background(self, images: List[np.ndarray]) -> np.ndarray:
        """Estimate static background as the per-pixel temporal median."""
        stack = np.stack([img.astype(np.float32) for img in images], axis=0)
        self._background = np.median(stack, axis=0)
        return self._background

    def mask_frame(self, image: np.ndarray) -> np.ndarray:
        if self._background is None:
            raise RuntimeError("Call fit_background(images) before mask_frame().")
        diff = np.abs(image.astype(np.float32) - self._background).mean(axis=2)
        mask = diff > self.threshold
        if self.dilate_px > 0:
            k = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
        return mask

    def mask_sequence(self, images: List[np.ndarray]) -> List[np.ndarray]:
        self.fit_background(images)
        return [self.mask_frame(img) for img in images]
