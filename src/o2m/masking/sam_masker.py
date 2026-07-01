"""Text-prompted segmentation masker (Grounded-SAM / SAM2).

Optional, higher-quality masker. Kept as a lazy-import stub so the core package
installs without the segmentation stack. Implement ``mask_frame`` against your
installed SAM variant; the contract is identical to ``ColorMotionMasker``
(True = foreground to exclude).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


class GroundedSAMMasker:
    def __init__(self, prompts: Sequence[str] = ("robot arm", "gripper", "bag"),
                 device: str = "cuda"):
        self.prompts = list(prompts)
        self.device = device
        self._model = None  # lazily constructed in _ensure_model()

    def _ensure_model(self):
        if self._model is None:
            raise NotImplementedError(
                "Wire up your Grounded-SAM / SAM2 checkpoint here. Install the "
                "`mask` extra and load the model into self._model."
            )
        return self._model

    def mask_frame(self, image: np.ndarray) -> np.ndarray:
        self._ensure_model()
        raise NotImplementedError

    def mask_sequence(self, images: List[np.ndarray]) -> List[np.ndarray]:
        return [self.mask_frame(img) for img in images]
