"""Pinhole camera model shared by splat rendering and robot overlay.

Holds intrinsics (fx, fy, cx, cy, w, h) and a camera-to-world pose (4x4) in the
splat/COLMAP world frame. Provides constructors from a transforms.json frame and
from intrinsic priors, plus an orbit generator for the static render mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..utils import geometry as geom


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    c2w: np.ndarray  # 4x4, OpenCV convention (x right, y down, z forward)

    @property
    def fovy_deg(self) -> float:
        return float(np.degrees(2 * np.arctan2(self.height / 2.0, self.fy)))

    @classmethod
    def from_intrinsics(cls, fx, fy, cx, cy, width, height,
                        c2w: np.ndarray | None = None) -> "Camera":
        return cls(fx, fy, cx, cy, int(width), int(height),
                   np.eye(4) if c2w is None else np.asarray(c2w, float))

    @classmethod
    def from_transforms_frame(cls, transforms: dict, frame_index: int = 0) -> "Camera":
        """Build from a Nerfstudio transforms dict (OpenGL c2w -> OpenCV here)."""
        f = transforms["frames"][frame_index]
        c2w_gl = np.array(f["transform_matrix"], dtype=float)
        c2w_cv = geom.opengl_c2w_to_opencv(c2w_gl)
        return cls(transforms["fl_x"], transforms["fl_y"],
                   transforms["cx"], transforms["cy"],
                   int(transforms["w"]), int(transforms["h"]), c2w_cv)

    def look_at(self, eye: np.ndarray, target: np.ndarray,
                up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> "Camera":
        """Return a copy whose pose looks from ``eye`` to ``target`` (OpenCV)."""
        eye = np.asarray(eye, float)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        Rm = np.stack([right, -true_up, forward], axis=1)  # OpenCV: y down, z fwd
        return Camera(self.fx, self.fy, self.cx, self.cy, self.width, self.height,
                      geom.make_se3(Rm, eye))


def interpolate_c2w(key_times: np.ndarray, key_c2w: np.ndarray,
                    query_times: np.ndarray) -> np.ndarray:
    """Interpolate OpenCV camera-to-world poses along a 1-D time axis.

    Rotations are slerp'd, translations linearly interpolated. Query times
    outside the key range are clamped to the nearest key. Used to render the
    splat at the dataset framerate (a pose for every recorded frame) from the
    sparser set of COLMAP-registered keyframes.

    Args:
        key_times: (K,) strictly increasing key timestamps/indices.
        key_c2w: (K, 4, 4) poses at the key times.
        query_times: (N,) timestamps/indices to interpolate at.
    Returns:
        (N, 4, 4) interpolated poses.
    """
    from scipy.spatial.transform import Rotation as R, Slerp

    key_times = np.asarray(key_times, float)
    qt = np.clip(np.asarray(query_times, float), key_times[0], key_times[-1])
    rots = R.from_matrix(key_c2w[:, :3, :3])
    slerp = Slerp(key_times, rots)
    out_rot = slerp(qt).as_matrix()
    out = np.repeat(np.eye(4)[None], len(qt), axis=0)
    out[:, :3, :3] = out_rot
    for k in range(3):
        out[:, k, 3] = np.interp(qt, key_times, key_c2w[:, k, 3])
    return out


def orbit_cameras(base: Camera, center: np.ndarray, radius: float,
                  n: int = 60, elevation: float = 0.3) -> List[Camera]:
    """Generate ``n`` cameras orbiting ``center`` for the static render mode."""
    cams: List[Camera] = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = center + np.array([radius * np.cos(a), radius * np.sin(a), radius * elevation])
        cams.append(base.look_at(eye, center))
    return cams
