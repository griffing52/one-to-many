"""Metric ZED scene geometry: depth for the clean plate in the robot base frame.

The ZED is calibrated into the base frame (``zed_extrinsic.npz``) but its
mono/video depth is only *relative*. This module fixes the metric scale with the
same trick the PnP calibration used: the **green gripper mount** is trackable in
the ZED image (HSV), and FK gives its metric position — hence its metric depth
along the ZED optical axis. Sampling frames across the episode gives (relative
disparity, metric depth) pairs at different distances, enough to fit the affine
``1/z ≈ a*disp + b`` that mono depth is invariant to.

The clean plate (robot inpainted out) is prepended to the SAME
Video-Depth-Anything sequence as the sampled real frames, so its disparity
shares their consistent scale and the fitted (a, b) applies to it directly.
The result is a **robot-free metric depth map** of the static scene, which is
what depth-ordered compositing and novel-view rendering need.

Outputs (via ``scripts/10_zed_metric_scene.py``):
  - ``align/zed_scene_metric.npz`` — plate depth (m) + K + c2w + fit diagnostics
  - ``pointcloud/zed_scene_metric.ply`` — the scene cloud in the base frame
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

# Green gripper mount in HSV (OpenCV ranges: H 0-179). Broad on purpose; the
# blob is the only saturated green in the scene.
GREEN_HSV_LO = np.array([40, 80, 60])
GREEN_HSV_HI = np.array([90, 255, 255])


@dataclass
class ScaleFit:
    a: float
    b: float
    n_samples: int
    n_inliers: int
    z_range: Tuple[float, float]
    rmse_m: float

    def apply(self, disp: np.ndarray) -> np.ndarray:
        """Consistent relative disparity -> metric depth (m). Invalid where
        the fitted inverse depth is non-positive."""
        inv = self.a * disp + self.b
        with np.errstate(divide="ignore"):
            z = 1.0 / inv
        z[inv <= 1e-6] = np.nan
        return z.astype(np.float32)


def track_green_gripper(frames_rgb: Sequence[np.ndarray], min_area: int = 150
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Centroid of the green gripper mount per frame.

    Returns (px (N,2) float [u,v], valid (N,) bool). Invalid when the blob is
    too small (gripper out of view / occluded).
    """
    import cv2
    px = np.full((len(frames_rgb), 2), np.nan)
    valid = np.zeros(len(frames_rgb), bool)
    for i, rgb in enumerate(frames_rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            continue
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[best, cv2.CC_STAT_AREA] < min_area:
            continue
        px[i] = cents[best]
        valid[i] = True
    return px, valid


def fit_disp_to_inv_depth(disp: np.ndarray, z_metric: np.ndarray,
                          iters: int = 3, inlier_frac: float = 0.8) -> ScaleFit:
    """Robust affine fit ``1/z = a*disp + b`` (trimmed least squares)."""
    d = np.asarray(disp, float)
    z = np.asarray(z_metric, float)
    keep = np.isfinite(d) & np.isfinite(z) & (z > 1e-3)
    d, z = d[keep], z[keep]
    if len(d) < 6:
        raise ValueError(f"Too few scale samples ({len(d)}); need >= 6.")
    inl = np.ones(len(d), bool)
    coef = None
    for _ in range(iters):
        A = np.stack([d[inl], np.ones(inl.sum())], 1)
        coef, *_ = np.linalg.lstsq(A, 1.0 / z[inl], rcond=None)
        pred_inv = coef[0] * d + coef[1]
        ok = pred_inv > 1e-6
        err = np.abs(np.where(ok, 1.0 / np.where(ok, pred_inv, 1.0), np.inf) - z)
        thresh = np.quantile(err[np.isfinite(err)], inlier_frac)
        inl = err <= max(thresh, 1e-3)
    pred_inv = coef[0] * d[inl] + coef[1]
    rmse = float(np.sqrt(np.mean((1.0 / pred_inv - z[inl]) ** 2)))
    fit = ScaleFit(float(coef[0]), float(coef[1]), len(d), int(inl.sum()),
                   (float(z.min()), float(z.max())), rmse)
    log.info("Scale fit: a=%.5f b=%.5f  %d/%d inliers, z in [%.2f, %.2f]m, "
             "RMSE %.3fm", fit.a, fit.b, fit.n_inliers, fit.n_samples,
             *fit.z_range, fit.rmse_m)
    return fit


def table_plane_samples(plate_disp: np.ndarray, plate_rgb: np.ndarray,
                        K: np.ndarray, c2w: np.ndarray,
                        exclude: Optional[np.ndarray] = None,
                        plane_z: float = 0.0,
                        z_range: Tuple[float, float] = (0.35, 2.2),
                        max_samples: int = 4000
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """(disparity, metric depth) pairs from the table plane.

    The robot base sits on the table, so the tabletop is the base-frame plane
    ``z = plane_z``; each white-cloth pixel's metric depth follows from the
    calibrated ray-plane intersection. This spans the full near-to-far depth
    range of the scene — a far stronger anchor for the affine disparity fit
    than the gripper track alone (whose z-spread is ~0.2 m).
    """
    import cv2
    H, W = plate_disp.shape
    hsv = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2HSV)
    whiteish = (hsv[..., 1] < 60) & (hsv[..., 2] > 120)
    if exclude is not None:
        whiteish &= ~exclude

    ys, xs = np.mgrid[0:H, 0:W]
    dirs = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1],
                     np.ones_like(plate_disp)], axis=-1)      # optic-axis depth 1
    denom = dirs @ c2w[:3, :3].T[:, 2]                        # (R @ dir)_z
    with np.errstate(divide="ignore", invalid="ignore"):
        z_plane = (plane_z - c2w[2, 3]) / denom
    good = whiteish & np.isfinite(z_plane) & (z_plane > z_range[0]) & (z_plane < z_range[1])
    d, z = plate_disp[good], z_plane[good]
    if len(d) > max_samples:
        sel = np.random.default_rng(0).choice(len(d), max_samples, replace=False)
        d, z = d[sel], z[sel]
    log.info("Table-plane anchors: %d pixels, z in [%.2f, %.2f]m",
             len(d), float(z.min()) if len(z) else np.nan,
             float(z.max()) if len(z) else np.nan)
    return d, z


def build_zed_metric_scene(zed_frame_paths: List[Path], joints: np.ndarray,
                           clean_plate: np.ndarray, K: np.ndarray,
                           c2w: np.ndarray, robot_model,
                           stride: int = 8, encoder: str = "vits",
                           patch: int = 2) -> Tuple[np.ndarray, ScaleFit, dict]:
    """Metric depth (m) for the clean plate, in the calibrated ZED view.

    The affine ``1/z = a*disp + b`` is fitted on **table-plane anchors** (wide
    depth range, hundreds of pixels) and **validated** on the FK-tracked green
    gripper (independent metric measurements on the robot).

    Args:
        zed_frame_paths: all recorded ZED frames (index-aligned with joints).
        joints: (N,6) measured joints (rad).
        clean_plate: HxWx3 RGB, robot inpainted out.
        K, c2w: ZED intrinsics + base-frame pose from ``zed_extrinsic.npz``.
        robot_model: :class:`o2m.robot.PiperModel` (ee_frame = green mount proxy).
        stride: frame sampling stride for the gripper validation track.
        patch: half-size of the disparity sampling window at the tracked pixel.

    Returns:
        (plate_depth_m HxW, ScaleFit, diagnostics dict)
    """
    import cv2
    from PIL import Image
    from ..depth import VideoDepthEstimator

    idx = list(range(0, len(zed_frame_paths), stride))
    frames = [np.asarray(Image.open(zed_frame_paths[i]).convert("RGB")) for i in idx]
    px, valid = track_green_gripper(frames)
    log.info("Gripper tracked in %d/%d sampled ZED frames", int(valid.sum()), len(idx))

    # Metric gripper depth along the ZED optical axis, from FK + calibration.
    w2c = np.linalg.inv(c2w)
    z_fk = np.full(len(idx), np.nan)
    uv_fk = np.full((len(idx), 2), np.nan)
    ee = robot_model.ee_frame
    for k, i in enumerate(idx):
        p_base = robot_model.fk(joints[i], [ee])[ee][:3, 3]
        p_cam = w2c[:3, :3] @ p_base + w2c[:3, 3]
        if p_cam[2] > 1e-3:
            z_fk[k] = p_cam[2]
            uv_fk[k] = (K[:2, :2] @ (p_cam[:2] / p_cam[2])) + K[:2, 2]
    d_px = np.linalg.norm(px - uv_fk, axis=1)
    use = valid & np.isfinite(z_fk) & (d_px < 120.0)

    # ONE consistent disparity stack: plate first, then the sampled frames.
    vde = VideoDepthEstimator(encoder=encoder)
    disp = vde.estimate_sequence([clean_plate] + frames)
    plate_disp, frame_disp = disp[0], disp[1:]

    d_grip = np.full(len(idx), np.nan)
    for k in np.nonzero(use)[0]:
        u, v = int(round(px[k, 0])), int(round(px[k, 1]))
        d_grip[k] = np.median(frame_disp[k][max(0, v - patch):v + patch + 1,
                                            max(0, u - patch):u + patch + 1])

    # Exclude the robot/inpaint-ghost region from the plane anchors: where the
    # plate differs from a real frame, the robot was inpainted out and the
    # plate's disparity there is unreliable.
    diff = np.abs(clean_plate.astype(int) - np.asarray(frames[0], int)).sum(-1)
    ghost = cv2.dilate((diff > 60).astype(np.uint8),
                       np.ones((25, 25), np.uint8)).astype(bool)

    d_pl, z_pl = table_plane_samples(plate_disp, clean_plate, K, c2w, exclude=ghost)
    fit = fit_disp_to_inv_depth(d_pl, z_pl)
    if fit.a <= 0:
        raise RuntimeError(
            f"Scale fit has non-physical slope a={fit.a:.4f} (disparity must "
            "increase with inverse depth) — check calibration / plane anchors.")

    # Independent validation on the gripper track.
    grip_rmse = float("nan")
    if use.any():
        z_pred = fit.apply(d_grip[use])
        grip_rmse = float(np.sqrt(np.nanmean((z_pred - z_fk[use]) ** 2)))
        log.info("Gripper validation: RMSE %.3fm over %d frames "
                 "(FK z in [%.2f, %.2f]m)", grip_rmse, int(use.sum()),
                 float(np.nanmin(z_fk[use])), float(np.nanmax(z_fk[use])))

    plate_depth = fit.apply(plate_disp)
    # The inpaint ghost has bogus (near) disparity -> it would wrongly occlude
    # the rendered robot. Replace its depth with the nearest real scene depth.
    plate_depth = _repair_region(plate_depth, ghost)
    diag = {
        "n_frames_sampled": len(idx),
        "n_tracked": int(valid.sum()),
        "n_gripper_used": int(use.sum()),
        "gripper_rmse_m": grip_rmse,
        "plate_depth_median_m": float(np.nanmedian(plate_depth)),
        "ghost_frac": float(ghost.mean()),
    }
    return plate_depth, fit, diag


def _repair_region(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overwrite ``mask`` pixels with the nearest unmasked depth (+ light blur)."""
    import cv2
    from scipy import ndimage
    if not mask.any():
        return depth
    idx = ndimage.distance_transform_edt(mask, return_distances=False,
                                         return_indices=True)
    out = depth[tuple(idx)]
    sm = cv2.GaussianBlur(out, (0, 0), 9)
    out = np.where(mask, sm, depth)
    return out.astype(np.float32)


def load_scene_depth(npz_path: str | Path) -> Optional[np.ndarray]:
    """Plate metric depth from ``zed_scene_metric.npz`` (None if missing)."""
    p = Path(npz_path)
    if not p.exists():
        return None
    return np.load(str(p))["depth"]
