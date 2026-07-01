"""Inverse kinematics for the Piper arm.

Damped least-squares Jacobian iteration, mirroring
``kinematic_translator.solve_piper_ik`` but operating on a :class:`PiperModel`
(no UR5e dependency). Use this only when an altered trajectory is specified in
EE space; to re-render the *original* trajectory, feed measured joints directly.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .urdf_model import PiperModel


def q_for_ee(model: PiperModel, target_se3: np.ndarray, q_guess: np.ndarray,
             max_iters: int = 300, tol: float = 1e-4, damping: float = 1e-4,
             step_size: float = 1.0) -> Tuple[np.ndarray, bool, int, float]:
    """Solve for joints reaching ``target_se3`` (4x4, base frame).

    Levenberg-Marquardt damped least squares: the damping adapts (grows when a
    step increases the error, shrinks when it decreases), which is stable near
    singularities / at full extension yet precise when the target is reachable.

    Returns (q, converged, iters, residual_norm). ``converged`` is True only if
    the residual reached ``tol`` — a large residual flags an infeasible target
    (e.g. base shifted too far to reach the grasp).
    """
    pin = model._pin
    target = pin.SE3(np.asarray(target_se3)[:3, :3].copy(),
                     np.asarray(target_se3)[:3, 3].copy())
    q = np.asarray(q_guess, dtype=float).copy().reshape(-1)
    if q.shape[0] != model.model.nq:
        q = model._pad_q(q)
    ee_id = model.model.getFrameId(model.ee_frame)

    def residual_at(qq):
        pin.forwardKinematics(model.model, model.data, qq)
        pin.updateFramePlacements(model.model, model.data)
        e = pin.log6(model.data.oMf[ee_id].actInv(target)).vector
        return e, float(np.linalg.norm(e))

    lam = damping
    err, residual = residual_at(q)
    for it in range(max_iters):
        if residual < tol:
            return q, True, it, residual
        J = pin.computeFrameJacobian(model.model, model.data, q, ee_id,
                                     pin.ReferenceFrame.LOCAL)
        # err = log6(current^-1 . target): twist (local frame) toward target.
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
        q_new = pin.integrate(model.model, q, step_size * dq)
        err_new, res_new = residual_at(q_new)
        if res_new < residual:           # accept, become less damped (Gauss-Newton)
            q, err, residual = q_new, err_new, res_new
            lam = max(lam * 0.5, 1e-8)
        else:                            # reject, damp more (gradient-descent-like)
            lam = min(lam * 4.0, 1e3)

    return q, False, max_iters, residual
