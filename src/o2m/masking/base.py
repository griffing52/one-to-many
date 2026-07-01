from __future__ import annotations

from typing import List, Protocol

import numpy as np


class Masker(Protocol):
    """True = dynamic foreground to exclude from reconstruction."""

    def mask_frame(self, image: np.ndarray) -> np.ndarray:  # (H,W) bool
        ...

    def mask_sequence(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Default: per-frame; subclasses may use temporal context."""
        ...
