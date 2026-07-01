"""Splat-world <-> robot-base alignment — the crux of the robot overlay.

The splat lives in COLMAP's arbitrary scale and frame; the robot's EE/joint data
lives in the robot base frame. To draw the URDF arm inside the splat we need a
similarity transform ``T_splat<-base`` (scale + rotation + translation).

We recover it from correspondences that exist for free: the wrist camera is
rigidly mounted on the arm, so for each frame i we have
  - C_i = COLMAP wrist-camera centre (splat frame), and
  - B_i = FK of the camera link at the measured joints (base frame).
With N (~hundreds) of frames, Umeyama on the camera centres solves for the
sim(3); the scale fixes COLMAP units to metres. See docs/alignment.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from ..utils import geometry as geom


@dataclass
class Sim3:
    s: float
    R: np.ndarray   # (3,3)
    t: np.ndarray   # (3,)

    def matrix(self) -> np.ndarray:
        return geom.sim3_to_matrix(self.s, self.R, self.t)

    def apply(self, pose_base: np.ndarray) -> np.ndarray:
        """Map a base-frame 4x4 pose into the splat frame.

        Translation scales by s; rotation composes with R (a similarity acting on
        a rigid pose keeps the rotation part special-orthogonal).
        """
        out = np.eye(4)
        out[:3, :3] = self.R @ pose_base[:3, :3]
        out[:3, 3] = self.s * (self.R @ pose_base[:3, 3]) + self.t
        return out

    def inv_apply(self, pose_splat: np.ndarray) -> np.ndarray:
        """Map a splat-frame 4x4 pose back into the (metric) base frame.

        Used to express the render viewpoint in the robot base frame so the URDF
        arm (metres, base frame) is rendered from the same physical viewpoint as
        the splat background.
        """
        out = np.eye(4)
        out[:3, :3] = self.R.T @ pose_splat[:3, :3]
        out[:3, 3] = self.R.T @ (pose_splat[:3, 3] - self.t) / self.s
        return out

    def to_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"s": self.s, "R": self.R.tolist(), "t": self.t.tolist()}, f, indent=2)
        return path

    @classmethod
    def from_json(cls, path: Path) -> "Sim3":
        with open(path) as f:
            d = json.load(f)
        return cls(float(d["s"]), np.array(d["R"]), np.array(d["t"]))


class WorldAligner:
    """Solve ``T_splat<-base`` from wrist-camera correspondences."""

    @staticmethod
    def from_wrist_fk(colmap_cam_centers: np.ndarray,
                      fk_cam_centers_base: np.ndarray) -> Tuple[Sim3, dict]:
        """Umeyama-fit base camera centres to COLMAP camera centres.

        Args:
            colmap_cam_centers: (N,3) wrist-camera centres in the splat frame.
            fk_cam_centers_base: (N,3) FK camera centres in the base frame,
                index-aligned to ``colmap_cam_centers``.

        Returns:
            (Sim3, diagnostics) where diagnostics includes residual RMS (metres).
        """
        src = np.asarray(fk_cam_centers_base, float)
        dst = np.asarray(colmap_cam_centers, float)
        if src.shape != dst.shape or src.shape[0] < 3:
            raise ValueError("Need >=3 index-aligned 3D correspondences.")

        s, R, t = geom.umeyama_sim3(src, dst, with_scale=True)
        sim3 = Sim3(s, R, t)

        pred = (s * (R @ src.T).T) + t
        resid = np.linalg.norm(pred - dst, axis=1)
        diag = {
            "n": int(src.shape[0]),
            "scale": s,
            "residual_rms": float(np.sqrt((resid ** 2).mean())),
            "residual_max": float(resid.max()),
        }
        return sim3, diag

    @staticmethod
    def from_known_points(base_points: np.ndarray,
                          splat_points: np.ndarray) -> Tuple[Sim3, dict]:
        """Fallback: align from manual correspondences (e.g. table corners)."""
        s, R, t = geom.umeyama_sim3(np.asarray(base_points, float),
                                    np.asarray(splat_points, float), with_scale=True)
        return Sim3(s, R, t), {"n": int(len(base_points)), "scale": s}
