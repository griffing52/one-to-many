"""Monocular depth estimation via Depth-Anything-V2 (HuggingFace)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


class MonoDepthEstimator:
    """Wraps the HF depth-estimation pipeline. Returns affine-invariant relative
    depth per image (aligned to metric later, against COLMAP points)."""

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: int | None = None):
        from transformers import pipeline
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("depth-estimation", model=model, device=device)

    def estimate(self, image) -> np.ndarray:
        """image: PIL.Image or HxWx3 RGB array -> HxW float32 relative depth."""
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        out = self.pipe(image)
        d = out["predicted_depth"]
        return np.asarray(d, dtype=np.float32)

    def estimate_paths(self, paths: List[Path]) -> List[np.ndarray]:
        from PIL import Image
        return [self.estimate(Image.open(p).convert("RGB")) for p in paths]
