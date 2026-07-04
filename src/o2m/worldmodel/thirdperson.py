"""Third-person (ZED) view synthesis: URDF robot composited over a clean plate.

The ZED is fixed and was calibrated into the robot BASE frame (target-free PnP on
the tracked green gripper, ~7px). So a perturbed joint config renders directly
from that calibrated viewpoint and composites over the real ZED "clean plate"
(the scene with the robot inpainted out). Bag fidelity in this view is coarse by
design -- the third-person view is for *what/where*, the wrist view is for *how*.

Inputs (all discoverable from ``configs/worldmodel.yaml``):
  - ``zed_extrinsic_npz``: dict with ``c2w`` (4x4, base frame) and ``K`` (3x3).
  - ``clean_plate``: RGB PNG (1280x720) background.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..splat.camera import Camera


def load_zed_camera(npz_path: str | Path) -> Camera:
    """Build the calibrated ZED :class:`Camera` (base frame) from the npz."""
    d = np.load(str(npz_path))
    K, c2w = d["K"], d["c2w"]
    h, w = 720, 1280
    return Camera.from_intrinsics(float(K[0, 0]), float(K[1, 1]),
                                  float(K[0, 2]), float(K[1, 2]), w, h, c2w)


class ThirdPersonRenderer:
    """Composite the Piper arm (at given joints) over the ZED clean plate.

    With ``scene_depth`` (metric plate depth from ``scripts/10_zed_metric_scene.py``)
    the composite is DEPTH-ORDERED: scene geometry nearer than the rendered arm
    (by more than ``depth_margin``, which absorbs the ~0.1 m metric-depth error)
    hides it — e.g. the arm passing behind the rack.
    """

    def __init__(self, robot_renderer, camera: Camera, clean_plate: np.ndarray,
                 scene_depth: np.ndarray | None = None, depth_margin: float = 0.12):
        self.renderer = robot_renderer
        self.camera = camera
        self.plate = clean_plate
        self.scene_depth = scene_depth
        self.depth_margin = depth_margin

    def render(self, q: np.ndarray) -> np.ndarray:
        """One RGB frame: arm at joints ``q`` over the clean plate."""
        from ..render.composite import composite_rgba_over
        fg, alpha, fg_depth = self.renderer.render_rgba(q, self.camera)
        if self.scene_depth is None:
            return composite_rgba_over(self.plate, fg, alpha)
        bg_depth = np.nan_to_num(self.scene_depth, nan=0.0)  # nan -> "far"
        return composite_rgba_over(self.plate, fg, alpha,
                                   fg_depth=fg_depth - self.depth_margin,
                                   bg_depth=bg_depth)
