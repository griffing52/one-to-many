"""FK-posed fusion of wrist video-depth into a static base-frame scene cloud.

The moving wrist camera sweeps the scene, and FK of ``hand_cam`` gives every
frame's camera pose in the base frame directly — no feature matching or ICP
(both were shown to fail on this low-texture scene). Video-Depth-Anything makes
the per-frame depths share ONE consistent relative scale, so a single global
affine ``1/z = a*disp + b`` metrifies the whole sequence. That affine is fitted
against the metric ZED scene (``scripts/10_zed_metric_scene.py``) projected into
each wrist view (poses known on both sides -> no matching needed).

This is the "multi-frame fusion" fix for the wrist disocclusion-spray limit,
and it feeds the merged cloud that novel third-person views render from.

Frames after the grasp move the bag; fuse the PRE-GRASP range for a static
scene (the default in ``scripts/11_fuse_wrist_cloud.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logging import get_logger
from .scene_cloud import ScaleFit, fit_disp_to_inv_depth
from .wrist_warp import OPTICAL_FROM_LINK, GripperMask, WristIntrinsics

log = get_logger(__name__)


def optical_c2w(T_link_base: np.ndarray) -> np.ndarray:
    """FK ``hand_cam`` link pose (base<-link) -> optical-frame c2w (OpenCV).

    ``v_opt = OPTICAL_FROM_LINK.T @ v_link`` (see ``base_offset_to_camera``),
    so the optical rotation is ``R_link @ OPTICAL_FROM_LINK``.
    """
    out = np.asarray(T_link_base, float).copy()
    out[:3, :3] = out[:3, :3] @ OPTICAL_FROM_LINK
    return out


def project_scene_depth(scene_xyz: np.ndarray, c2w_opt: np.ndarray,
                        intr: WristIntrinsics, grid: int = 4
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the base-frame scene cloud into a wrist view -> sparse depth.

    Returns (us, vs, zs) at ``grid``-bucketed pixels, keeping the NEAREST point
    per bucket (approximates the visible surface of the point set).
    """
    w2c = np.linalg.inv(c2w_opt)
    p = scene_xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    ok = z > 0.05
    u = p[ok, 0] / z[ok] * intr.fx + intr.cx
    v = p[ok, 1] / z[ok] * intr.fy + intr.cy
    z = z[ok]
    inb = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    u, v, z = u[inb], v[inb], z[inb]
    # z-buffer per grid bucket
    gu, gv = (u // grid).astype(int), (v // grid).astype(int)
    key = gv * (intr.width // grid + 1) + gu
    order = np.argsort(z)                      # nearest first
    key, u, v, z = key[order], u[order], v[order], z[order]
    _, first = np.unique(key, return_index=True)
    return u[first], v[first], z[first]


def fit_wrist_scale(disps: np.ndarray, c2ws: Sequence[np.ndarray],
                    intr: WristIntrinsics, scene_xyz: np.ndarray,
                    gmask: Optional[np.ndarray] = None,
                    max_pairs_per_frame: int = 800) -> ScaleFit:
    """Global affine for the wrist disparity stack, anchored to the ZED scene.

    Args:
        disps: (N,H,W) consistent wrist disparity (Video-Depth-Anything).
        c2ws: N optical c2w poses (base frame, from FK).
        intr: wrist intrinsics.
        scene_xyz: (M,3) metric base-frame scene points (ZED cloud).
        gmask: HxW bool, True where the gripper occludes the scene (excluded).
    """
    rng = np.random.default_rng(0)
    d_all, z_all = [], []
    for disp, c2w in zip(disps, c2ws):
        us, vs, zs = project_scene_depth(scene_xyz, c2w, intr)
        iu, iv = us.astype(int), vs.astype(int)
        keep = np.ones(len(iu), bool)
        if gmask is not None:
            keep &= ~gmask[iv, iu]
        iu, iv, zs = iu[keep], iv[keep], zs[keep]
        if len(iu) > max_pairs_per_frame:
            sel = rng.choice(len(iu), max_pairs_per_frame, replace=False)
            iu, iv, zs = iu[sel], iv[sel], zs[sel]
        d_all.append(disp[iv, iu])
        z_all.append(zs)
    d_all = np.concatenate(d_all)
    z_all = np.concatenate(z_all)
    log.info("Wrist scale: %d (disparity, ZED-depth) pairs over %d frames",
             len(d_all), len(disps))
    fit = fit_disp_to_inv_depth(d_all, z_all, inlier_frac=0.7)
    if fit.a <= 0:
        raise RuntimeError(f"Wrist scale fit non-physical (a={fit.a:.4f}).")
    return fit


def fit_wrist_scale_triangulated(frames: Sequence[np.ndarray], disps: np.ndarray,
                                 c2ws: Sequence[np.ndarray], intr: WristIntrinsics,
                                 gmask: Optional[np.ndarray] = None,
                                 min_baseline: float = 0.015,
                                 max_reproj_px: float = 1.5) -> ScaleFit:
    """Global wrist affine from FK-metric two-view triangulation (no ZED needed).

    KLT-tracks corners between wrist frame pairs, triangulates each track with
    the FK relative pose (exact, metric), and fits ``1/z = a*disp + b`` on the
    (disparity, triangulated depth) pairs. Feature tracking between *adjacent*
    wrist frames is reliable on this scene (unlike wide-baseline matching), and
    the FK baseline supplies the metric scale.
    """
    import cv2
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
    border = 12
    d_all, z_all = [], []
    n_pairs = 0
    for i in range(len(frames)):
        # nearest later frame with enough FK baseline
        j = next((j for j in range(i + 1, min(i + 12, len(frames)))
                  if np.linalg.norm(c2ws[j][:3, 3] - c2ws[i][:3, 3]) >= min_baseline),
                 None)
        if j is None:
            continue
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(frames[j], cv2.COLOR_RGB2GRAY)
        fmask = np.full(g0.shape, 255, np.uint8)
        fmask[:border] = fmask[-border:] = 0
        fmask[:, :border] = fmask[:, -border:] = 0
        # The gripper prongs (rigid to the camera) reach into the bottom
        # corners OUTSIDE the trapezoid; camera-rigid features have ~zero flow
        # and triangulate to garbage. Exclude the whole bottom band.
        fmask[int(0.72 * g0.shape[0]):] = 0
        if gmask is not None:
            fmask[gmask] = 0
        p0 = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01,
                                     minDistance=8, mask=fmask)
        if p0 is None or len(p0) < 10:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None,
                                             winSize=(21, 21), maxLevel=3)
        ok = st.reshape(-1) == 1
        # Camera-rigid points show ~zero flow despite the baseline -> drop.
        ok &= np.linalg.norm((p1 - p0).reshape(-1, 2), axis=1) > 2.0
        p0v, p1v = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        if len(p0v) < 10:
            continue

        T_ji = np.linalg.inv(c2ws[j]) @ c2ws[i]         # cam_i -> cam_j
        P0 = K @ np.eye(4)[:3]
        P1 = K @ T_ji[:3]
        X = cv2.triangulatePoints(P0, P1, p0v.T, p1v.T)
        X = (X[:3] / X[3]).T                            # cam_i frame
        z = X[:, 2]
        Xj = X @ T_ji[:3, :3].T + T_ji[:3, 3]
        # filters: in front of both cams, low reprojection error in both views
        good = (z > 0.05) & (z < 3.0) & (Xj[:, 2] > 0.05)
        r0 = (X[:, :2] / X[:, 2:3]) * [intr.fx, intr.fy] + [intr.cx, intr.cy]
        r1 = (Xj[:, :2] / Xj[:, 2:3]) * [intr.fx, intr.fy] + [intr.cx, intr.cy]
        good &= (np.linalg.norm(r0 - p0v, axis=1) < max_reproj_px)
        good &= (np.linalg.norm(r1 - p1v, axis=1) < max_reproj_px)
        # Parallax gate: rays must subtend a real angle or depth is unstable.
        o_i, o_j = c2ws[i][:3, 3], c2ws[j][:3, 3]
        Xw = X @ c2ws[i][:3, :3].T + o_i
        r_i = Xw - o_i
        r_j = Xw - o_j
        cosang = np.einsum("nd,nd->n", r_i, r_j) / (
            np.linalg.norm(r_i, axis=1) * np.linalg.norm(r_j, axis=1) + 1e-12)
        good &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) > 0.4
        if not good.any():
            continue
        u = np.clip(np.round(p0v[good, 0]).astype(int), 0, intr.width - 1)
        v = np.clip(np.round(p0v[good, 1]).astype(int), 0, intr.height - 1)
        d_all.append(disps[i][v, u])
        z_all.append(z[good])
        n_pairs += 1

    if not d_all:
        raise RuntimeError("No triangulated scale anchors (tracking failed).")
    d_all = np.concatenate(d_all)
    z_all = np.concatenate(z_all)
    log.info("Wrist scale (triangulated): %d anchors from %d frame pairs, "
             "z in [%.2f, %.2f]m", len(d_all), n_pairs,
             float(np.percentile(z_all, 2)), float(np.percentile(z_all, 98)))
    fit = fit_disp_to_inv_depth(d_all, z_all, inlier_frac=0.7)
    if fit.a <= 0:
        raise RuntimeError(f"Wrist scale fit non-physical (a={fit.a:.4f}).")
    return fit


def align_clouds_icp(src_xyz: np.ndarray, dst_xyz: np.ndarray,
                     iters: int = 25, trim: float = 0.7,
                     corr_start: float = 0.30, corr_end: float = 0.04,
                     bbox_margin: float = 0.10, sample: int = 20000
                     ) -> Tuple[np.ndarray, dict]:
    """Rigid trimmed ICP: transform mapping ``src`` onto ``dst`` (both metric).

    Used to close the residual offset between the ZED cloud (whose base-frame
    pose carries the PnP/f-prior calibration error) and the FK-metric wrist
    cloud, in their overlap (rack front, table). Only ``src`` points inside the
    ``dst`` bounding box (+margin) participate — the rest of the scene (walls)
    has no counterpart. The correspondence radius anneals ``corr_start`` ->
    ``corr_end`` so a coarse initial offset can still converge tightly.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    lo, hi = dst_xyz.min(0) - bbox_margin, dst_xyz.max(0) + bbox_margin
    in_box = np.all((src_xyz >= lo) & (src_xyz <= hi), axis=1)
    src_pool = src_xyz[in_box]
    log.info("ICP: %d/%d src points inside the dst bbox", len(src_pool), len(src_xyz))
    if len(src_pool) < 500:
        log.warning("ICP: too little bbox overlap — keeping identity.")
        return np.eye(4), {"rms_before": float("nan"), "rms_after": float("nan"),
                           "n_pairs": 0, "translation_m": 0.0}
    src = src_pool[rng.choice(len(src_pool), min(sample, len(src_pool)), replace=False)]
    tree = cKDTree(dst_xyz[rng.choice(len(dst_xyz), min(4 * sample, len(dst_xyz)),
                                      replace=False)])
    T = np.eye(4)
    rms0 = None
    for it in range(iters):
        max_corr = corr_start * (corr_end / corr_start) ** (it / max(1, iters - 1))
        cur = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(cur, workers=-1)
        keep = d < max_corr
        if keep.sum() < 100:
            continue
        d_k = d[keep]
        cut = np.quantile(d_k, trim)
        keep[keep] &= d_k <= cut
        if rms0 is None:
            rms0 = float(np.sqrt((d[keep] ** 2).mean()))
        p = cur[keep]
        q = tree.data[j[keep]]
        pc, qc = p.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((p - pc).T @ (q - qc))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        t = qc - R @ pc
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        T = step @ T
    cur = src @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(cur, workers=-1)
    keep = d < corr_end
    if rms0 is None or keep.sum() < 100:
        log.warning("ICP: never found enough pairs — keeping identity.")
        return np.eye(4), {"rms_before": rms0 or float("nan"),
                           "rms_after": float("nan"), "n_pairs": int(keep.sum()),
                           "translation_m": 0.0}
    diag = {"rms_before": rms0,
            "rms_after": float(np.sqrt((d[keep] ** 2).mean())),
            "n_pairs": int(keep.sum()),
            "translation_m": float(np.linalg.norm(T[:3, 3]))}
    log.info("ICP ZED->wrist: overlap RMS %.3f -> %.3fm, |t|=%.3fm (%d pairs)",
             diag["rms_before"], diag["rms_after"], diag["translation_m"],
             diag["n_pairs"])
    return T, diag


def fuse_wrist_cloud(frames: Sequence[np.ndarray], depths: Sequence[np.ndarray],
                     c2ws: Sequence[np.ndarray], intr: WristIntrinsics,
                     gmask: Optional[np.ndarray] = None,
                     pixel_stride: int = 3, border: int = 12,
                     z_range: Tuple[float, float] = (0.12, 2.5),
                     voxel: float = 0.004, max_points: int = 1_500_000
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject metric wrist depths with FK poses and voxel-fuse.

    Returns (xyz (P,3) base frame, rgb (P,3) uint8).
    """
    ys, xs = np.mgrid[0:intr.height:pixel_stride, 0:intr.width:pixel_stride]
    sub_g = gmask[::pixel_stride, ::pixel_stride] if gmask is not None else None
    inner = (xs >= border) & (xs < intr.width - border) & \
            (ys >= border) & (ys < intr.height - border)

    all_xyz, all_rgb = [], []
    for rgb, depth, c2w in zip(frames, depths, c2ws):
        z = depth[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(z) & (z > z_range[0]) & (z < z_range[1]) & inner
        if sub_g is not None:
            valid &= ~sub_g
        zc = z[valid]
        xc = (xs[valid] - intr.cx) / intr.fx * zc
        yc = (ys[valid] - intr.cy) / intr.fy * zc
        pts = np.stack([xc, yc, zc], 1) @ c2w[:3, :3].T + c2w[:3, 3]
        all_xyz.append(pts)
        all_rgb.append(rgb[::pixel_stride, ::pixel_stride][valid])

    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    # Robust bbox clip (stray sky/blur points), then voxel-fuse.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    pad = 0.25 * (hi - lo)
    keep = np.all((xyz >= lo - pad) & (xyz <= hi + pad), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    from ..depth.dense_init import _voxel_downsample
    xyz, rgb = _voxel_downsample(xyz, rgb, voxel, max_points)
    log.info("Fused wrist cloud: %d points (voxel %.3fm, %d frames)",
             len(xyz), voxel, len(frames))
    return xyz, rgb
