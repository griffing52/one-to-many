"""Edit an EE trajectory to produce the "many" novel trajectories.

Operates on :class:`EETrajectory` in the robot base frame (metres / rotvec), the
same representation produced by the data loader and consumed by IK before
rendering.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..data.types import EETrajectory


class TrajectoryEditor:
    def __init__(self, traj: EETrajectory):
        self.traj = traj.copy()

    def translate(self, dxyz) -> "TrajectoryEditor":
        self.traj.positions = self.traj.positions + np.asarray(dxyz, float)
        return self

    def scale_about_start(self, factor: float) -> "TrajectoryEditor":
        p0 = self.traj.positions[0]
        self.traj.positions = p0 + (self.traj.positions - p0) * float(factor)
        return self

    def rotate_about_start(self, rpy_deg) -> "TrajectoryEditor":
        Rm = R.from_euler("xyz", np.radians(rpy_deg)).as_matrix()
        p0 = self.traj.positions[0]
        self.traj.positions = p0 + (self.traj.positions - p0) @ Rm.T
        rot = R.from_matrix(Rm) * R.from_rotvec(self.traj.rotvecs)
        self.traj.rotvecs = rot.as_rotvec()
        return self

    def lift(self, dz: float) -> "TrajectoryEditor":
        self.traj.positions[:, 2] += float(dz)
        return self

    def set_gripper(self, width_m: float) -> "TrajectoryEditor":
        self.traj.gripper[:] = float(width_m)
        return self

    def time_warp(self, factor: float) -> "TrajectoryEditor":
        """Resample to ``factor`` x the original length (>1 slower/longer)."""
        n_old = len(self.traj)
        n_new = max(2, int(round(n_old * factor)))
        xs_old = np.linspace(0, 1, n_old)
        xs_new = np.linspace(0, 1, n_new)
        def interp(a):
            return np.stack([np.interp(xs_new, xs_old, a[:, k]) for k in range(a.shape[1])], axis=1)
        self.traj = EETrajectory(
            timestamps=np.interp(xs_new, xs_old, self.traj.timestamps),
            positions=interp(self.traj.positions),
            rotvecs=interp(self.traj.rotvecs),
            gripper=np.interp(xs_new, xs_old, self.traj.gripper),
        )
        return self

    def result(self) -> EETrajectory:
        return self.traj
