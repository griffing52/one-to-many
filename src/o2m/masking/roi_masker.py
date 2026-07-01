"""Static region-of-interest masker for a rigidly-mounted gripper.

Because the wrist camera and gripper move together, the gripper (and a grasped
object) occupy a roughly *fixed* region of the wrist image every frame. Masking
that region removes the harmful false-static features that bias COLMAP poses and
the floaters they create in the splat — without any learned model.

Define the region as a normalized polygon (coords in [0,1], origin top-left).
The default is a bottom-centre trapezoid, where the Piper gripper appears.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

# Bottom-centre trapezoid (x, y) in normalized image coords.
DEFAULT_POLYGON: List[Tuple[float, float]] = [
    (0.30, 1.00), (0.70, 1.00), (0.62, 0.62), (0.38, 0.62),
]


class ROIMasker:
    def __init__(self, polygon: Sequence[Tuple[float, float]] = DEFAULT_POLYGON,
                 dilate_px: int = 5):
        self.polygon = list(polygon)
        self.dilate_px = dilate_px

    def mask_frame(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pts = np.array([[int(x * w), int(y * h)] for x, y in self.polygon], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        if self.dilate_px > 0:
            k = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            mask = cv2.dilate(mask, k)
        return mask.astype(bool)

    def mask_sequence(self, images: List[np.ndarray]) -> List[np.ndarray]:
        # Static region -> same mask for every frame.
        return [self.mask_frame(img) for img in images]
