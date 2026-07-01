"""Composite a rendered robot (RGBA) over a splat render, with optional
depth-aware occlusion so background splats in front of the arm stay visible.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def composite_rgba_over(bg_rgb: np.ndarray, fg_rgb: np.ndarray, fg_alpha: np.ndarray,
                        fg_depth: Optional[np.ndarray] = None,
                        bg_depth: Optional[np.ndarray] = None) -> np.ndarray:
    """Alpha-over ``fg`` onto ``bg``.

    Args:
        bg_rgb: (H,W,3) background (splat render).
        fg_rgb: (H,W,3) foreground (robot render).
        fg_alpha: (H,W) bool/float mask of foreground coverage.
        fg_depth, bg_depth: optional (H,W) depths; where bg is nearer than fg the
            foreground is hidden (occluded by reconstructed geometry).
    """
    bg = bg_rgb.astype(np.float32)
    fg = fg_rgb.astype(np.float32)
    a = fg_alpha.astype(np.float32)
    if a.ndim == 3:
        a = a[..., 0]

    if fg_depth is not None and bg_depth is not None:
        visible = bg_depth <= 0  # no background depth -> treat as far
        visible |= fg_depth <= bg_depth
        a = a * visible.astype(np.float32)

    a = a[..., None]
    out = a * fg + (1.0 - a) * bg
    return np.clip(out, 0, 255).astype(np.uint8)
