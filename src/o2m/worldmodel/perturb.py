"""Trajectory perturbation for the cheap world model.

A single recorded demonstration is turned into a *variation* by adding a smooth,
time-varying offset ``Delta(t)`` to the end-effector position (base frame). The
offset **converges to zero at the grasp frame** so the perturbed trajectory still
reaches the *same* grasp pose (the gripper actually closes on the bag), and stays
zero afterwards so the transport/place phase "ends about the same way". This is
the "shifted differently at every point, but reaches the correct spot" behaviour.

Feasibility is checked with IK: if every perturbed pose is reachable the variation
is a valid augmentation (label ``success``); if some poses are unreachable it is a
labelled failure (the a2l-pr "too far to grasp" signal).

See ``configs/worldmodel.yaml`` (``worldmodel.perturb``) for the tunable knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..data.types import EETrajectory


def smootherstep(x: np.ndarray) -> np.ndarray:
    """Ken Perlin's smootherstep on [0,1] -> [0,1] (zero 1st/2nd deriv at ends)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def detect_grasp_frame(gripper_width: np.ndarray) -> int:
    """First frame the gripper *closes* on the object after being open.

    The gripper opens during the reach and snaps shut at the grasp; we return the
    closing edge (first frame below 50% of the max opening, after the max-open
    frame). Used when ``grasp_frame`` is left null in the config.
    """
    g = np.asarray(gripper_width, float)
    open_frame = int(np.argmax(g))
    thr = 0.5 * float(g.max())
    after = np.where(g[open_frame:] < thr)[0]
    return int(open_frame + after[0]) if len(after) else int(len(g) // 2)


@dataclass
class PerturbationSpec:
    """One perturbation. ``base_offset`` is the peak EE offset (m, base frame)."""
    base_offset: Tuple[float, float, float] = (0.0, 0.04, 0.02)
    envelope: str = "converge_at_grasp"   # converge_at_grasp | constant
    grasp_frame: Optional[int] = None     # None -> auto-detect from gripper
    name: str = "perturb"


def offset_envelope(n_frames: int, grasp_frame: int, kind: str) -> np.ndarray:
    """Per-frame scalar weight w(t) in [0,1] multiplying ``base_offset``.

    - ``converge_at_grasp``: w=1 at t=0, smoothly -> 0 at the grasp frame, and 0
      afterwards. The approach is offset; the grasp and everything after it match
      the original (so it reaches the bag and ends the same way).
    - ``constant``: w=1 everywhere (a rigid base shift; useful for pure failure
      simulation where the arm never corrects).
    """
    t = np.arange(n_frames)
    if kind == "constant":
        return np.ones(n_frames)
    if kind == "converge_at_grasp":
        gf = max(1, int(grasp_frame))
        w = smootherstep((gf - t) / gf)   # 1 at t=0 -> 0 at t=gf
        w[t >= gf] = 0.0
        return w
    raise ValueError(f"Unknown envelope {kind!r}")


@dataclass
class PerturbedTrajectory:
    """Result of :func:`perturb_trajectory`."""
    traj: EETrajectory                 # perturbed EE trajectory (base frame)
    offsets: np.ndarray                # (N,3) Delta(t) in the base frame (m)
    weights: np.ndarray               # (N,) envelope w(t)
    grasp_frame: int
    spec: PerturbationSpec


def perturb_trajectory(traj: EETrajectory, spec: PerturbationSpec) -> PerturbedTrajectory:
    """Add the converging offset to ``traj`` (orientation & gripper unchanged)."""
    n = len(traj)
    gf = spec.grasp_frame
    if gf is None:
        gf = detect_grasp_frame(traj.gripper)
    w = offset_envelope(n, gf, spec.envelope)
    offsets = w[:, None] * np.asarray(spec.base_offset, float)[None, :]
    out = traj.copy()
    out.positions = out.positions + offsets
    return PerturbedTrajectory(traj=out, offsets=offsets, weights=w,
                               grasp_frame=int(gf), spec=spec)


@dataclass
class FeasibilityReport:
    """IK feasibility of a perturbed trajectory."""
    reachable: np.ndarray              # (N,) bool per frame
    residuals: np.ndarray             # (N,) IK residual norm (m/rad)
    joints: np.ndarray                # (N, nq) IK solution per frame
    success: bool                     # all frames reachable
    max_residual: float
    n_unreachable: int


def check_feasibility(model, pert: PerturbedTrajectory, measured_joints: np.ndarray,
                      tol: float = 5e-3) -> FeasibilityReport:
    """Solve IK for every perturbed pose; label success/failure.

    ``model`` is a :class:`o2m.robot.PiperModel``. The IK **target is the measured
    arm's FK pose translated by the base-frame offset**, warm-started at the
    measured joints. This is deliberate: the recorded ``slave_ee_*`` stream is in a
    different tool convention than the URDF ``link6`` FK, so IK-ing the recorded
    pose directly lands on the wrong branch (a wrong-looking arm). Anchoring to
    FK(measured) means a zero offset returns the measured joints exactly, and the
    rendered arm always matches the real one plus the intended shift.
    """
    from ..robot.ik import q_for_ee

    n = len(pert.traj)
    ee_id = model.ee_frame
    joints, residuals = [], []
    for i in range(n):
        q_meas = measured_joints[i]
        target = model.fk(q_meas, [ee_id])[ee_id].copy()
        target[:3, 3] = target[:3, 3] + pert.offsets[i]   # shift in the base frame
        q, _conv, _it, res = q_for_ee(model, target, q_meas, tol=tol)
        joints.append(q.copy())
        residuals.append(res)
    residuals = np.asarray(residuals)
    reachable = residuals < tol
    return FeasibilityReport(
        reachable=reachable, residuals=residuals, joints=np.stack(joints),
        success=bool(reachable.all()), max_residual=float(residuals.max()),
        n_unreachable=int((~reachable).sum()))
