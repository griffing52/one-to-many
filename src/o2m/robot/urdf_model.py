"""Piper arm model: Pinocchio for kinematics, MuJoCo for rendering.

Kinematics mirror the existing ``kinematic_translator`` utilities (same
Pinocchio FK and the damped-least-squares IK used there); the MuJoCo model is
built with the shared ``cross_embodiment.mujoco_scene`` URDF loader so meshes
resolve identically to the rest of the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ._external import ensure_on_path

_EE_CANDIDATES = ["tool0", "ee_link", "flange", "tcp", "gripper", "link6", "end_effector"]
_CAM_CANDIDATES = ["camera_link", "camera", "wrist_camera", "camera_color_frame", "rgb_camera"]


class PiperModel:
    def __init__(self, urdf_path: str | Path, urdf_dir: str | Path,
                 base_frame: str = "base_link",
                 ee_frame: Optional[str] = None,
                 camera_frame: Optional[str] = None):
        import pinocchio as pin

        self._pin = pin
        self.urdf_path = Path(urdf_path)
        self.urdf_dir = Path(urdf_dir)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        names = [f.name for f in self.model.frames]
        self.base_frame = base_frame if base_frame in names else names[0]
        self.ee_frame = ee_frame or self._guess(names, _EE_CANDIDATES)
        self.camera_frame = camera_frame or self._guess(names, _CAM_CANDIDATES, required=False)

    @staticmethod
    def _guess(names: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
        for c in candidates:
            if c in names:
                return c
        for name in reversed(names):
            if name.lower() not in {"universe", "world"}:
                return name if required else None
        return None

    @property
    def nq(self) -> int:
        return self.model.nq

    def fk(self, q: np.ndarray, frames: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Forward kinematics in the base frame -> {frame_name: 4x4}."""
        pin = self._pin
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.model.nq:
            q = self._pad_q(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        frames = frames or [f for f in (self.base_frame, self.ee_frame, self.camera_frame) if f]
        out: Dict[str, np.ndarray] = {}
        for name in frames:
            fid = self.model.getFrameId(name)
            T = self.data.oMf[fid]
            M = np.eye(4)
            M[:3, :3] = T.rotation
            M[:3, 3] = T.translation
            out[name] = M
        return out

    def _pad_q(self, q: np.ndarray) -> np.ndarray:
        full = np.zeros(self.model.nq)
        full[:min(len(q), self.model.nq)] = q[:self.model.nq]
        return full

    def camera_pose_base(self, q: np.ndarray) -> np.ndarray:
        """Wrist-camera pose in the base frame (used for sim3 alignment)."""
        if not self.camera_frame:
            raise RuntimeError("No camera frame in URDF; use the *_camera URDF variant.")
        return self.fk(q, [self.camera_frame])[self.camera_frame]

    # --- MuJoCo model for rendering ---------------------------------------
    def build_mujoco(self):
        """Compile a single-arm MuJoCo model with resolved meshes."""
        import mujoco
        ensure_on_path()
        from cross_embodiment.mujoco_scene import load_urdf_with_assets

        xml, assets = load_urdf_with_assets(self.urdf_dir, self.urdf_path)
        return mujoco.MjModel.from_xml_string(xml, assets)
