"""SE(3) / Sim(3) geometry helpers.

Pure numpy + scipy so the data/alignment/render-math path has no hard
dependency on pinocchio. Conventions:

- A pose is a 4x4 homogeneous matrix (camera/link-to-world unless noted).
- Rotation vectors follow scipy ``Rotation`` (axis-angle, xyz Euler convention).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def make_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def se3_from_rotvec(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return make_se3(R.from_rotvec(np.asarray(rotvec, float)).as_matrix(), translation)


def se3_from_euler(euler_xyz: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return make_se3(R.from_euler("xyz", np.asarray(euler_xyz, float)).as_matrix(), translation)


def invert_se3(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = Rm.T
    out[:3, 3] = -Rm.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    pts = np.asarray(pts, dtype=float)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


# OpenCV (x right, y down, z forward) <-> OpenGL (x right, y up, z backward):
# flip the camera's local y and z axes.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def opencv_c2w_to_opengl(c2w_cv: np.ndarray) -> np.ndarray:
    return c2w_cv @ CV_TO_GL


def opengl_c2w_to_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    return c2w_gl @ CV_TO_GL  # involution: its own inverse


def umeyama_sim3(src: np.ndarray, dst: np.ndarray,
                 with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping ``src`` -> ``dst`` (Umeyama 1991).

    Args:
        src: (N, 3) source points.
        dst: (N, 3) destination points.
        with_scale: estimate a uniform scale (set False for rigid SE3).

    Returns:
        (s, Rm, t) such that ``dst ~= s * Rm @ src + t``.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    Rm = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float(np.trace(np.diag(D) @ S) / var_src)
    else:
        s = 1.0

    t = mu_dst - s * Rm @ mu_src
    return s, Rm, t


def sim3_to_matrix(s: float, Rm: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack (s, R, t) into a single 4x4 matrix (scale folded into rotation block)."""
    T = np.eye(4)
    T[:3, :3] = s * Rm
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T
