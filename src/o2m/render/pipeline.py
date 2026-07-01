"""Configurable render pipeline.

Modes (selected via configs/pipeline.yaml render.mode):

- ``static``:        fly a free/orbit camera through the environment splat.
- ``robot_overlay``: render the splat from a fixed viewpoint, render the Piper
                     arm (URDF+FK at edited EE poses, mapped base->splat via the
                     sim3), and composite. This is the main deliverable.
- ``dynamic``:       robot_overlay plus a re-posed manipulated object (stretch).

The class is deliberately render-backend agnostic: it takes already-loaded
``SplatModel`` / ``RobotRenderer`` / ``Sim3`` objects so it can be unit-tested
with stubs and so the heavy splat/mujoco imports stay at the call site.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..align import Sim3
from ..data.types import EETrajectory
from ..splat.camera import Camera
from .composite import composite_rgba_over


class RenderPipeline:
    def __init__(self, splat_model, robot_renderer=None, robot_model=None,
                 sim3: Optional[Sim3] = None):
        self.splat = splat_model
        self.robot_renderer = robot_renderer
        self.robot_model = robot_model
        self.sim3 = sim3

    # --- modes -------------------------------------------------------------
    def render_static(self, cameras: List[Camera]) -> List[np.ndarray]:
        frames = []
        for cam in cameras:
            rgb, _, _ = self.splat.render(cam)
            frames.append(rgb)
        return frames

    def render_robot_overlay(self, viewpoint: Camera, joints_seq: np.ndarray
                             ) -> List[np.ndarray]:
        """Composite the arm (one config per timestep) over a fixed viewpoint.

        Args:
            viewpoint: fixed render camera, pose already in the splat frame.
            joints_seq: (T, nq) joint configs to render (measured or from IK).
        """
        if self.robot_renderer is None:
            raise RuntimeError("robot_overlay needs a RobotRenderer.")
        bg_rgb, bg_depth, _ = self.splat.render(viewpoint)
        frames = []
        for q in joints_seq:
            fg_rgb, fg_alpha, fg_depth = self.robot_renderer.render_rgba(q, viewpoint)
            frames.append(composite_rgba_over(bg_rgb, fg_rgb, fg_alpha,
                                              fg_depth=fg_depth, bg_depth=bg_depth))
        return frames

    def render_dynamic(self, viewpoint: Camera, joints_seq: np.ndarray,
                       object_renderer=None, object_poses=None) -> List[np.ndarray]:
        """Stretch: robot_overlay plus a re-posed object. See docs/stretch_dynamic.md."""
        frames = self.render_robot_overlay(viewpoint, joints_seq)
        if object_renderer is None:
            return frames
        out = []
        for frame, pose in zip(frames, object_poses):
            o_rgb, o_alpha, o_depth = object_renderer.render_rgba(pose, viewpoint)
            out.append(composite_rgba_over(frame, o_rgb, o_alpha, fg_depth=o_depth))
        return out

    # --- helpers -----------------------------------------------------------
    def ee_traj_to_joints(self, traj: EETrajectory,
                          measured_joints: Optional[np.ndarray] = None) -> np.ndarray:
        """Joints for a trajectory: measured if given (replay), else IK per pose."""
        if measured_joints is not None:
            return measured_joints
        if self.robot_model is None:
            raise RuntimeError("Need a PiperModel for IK when measured joints are absent.")
        from ..robot.ik import q_for_ee
        q = np.zeros(self.robot_model.nq)
        out = []
        for i in range(len(traj)):
            q, *_ = q_for_ee(self.robot_model, traj.se3_at(i), q)
            out.append(q.copy())
        return np.stack(out, axis=0)
