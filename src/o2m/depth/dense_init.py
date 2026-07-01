"""Build a dense seed point cloud for splatfacto from monocular depth.

For each registered frame we:
  1. read the COLMAP sparse points visible in it -> (pixel, metric depth) anchors;
  2. robustly fit mono-depth -> metric depth (best of direct / disparity model);
  3. unproject the dense mono-depth map into world points (COLMAP/splat frame);
  4. aggregate across frames and voxel-downsample.

The result is written as a coloured .ply that splatfacto initialises from, giving
correct geometry on textureless surfaces that photometric SfM leaves empty.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils import get_logger

log = get_logger("o2m.depth")


def _intrinsics(cam) -> Tuple[float, float, float, float]:
    p = np.asarray(cam.params, float)
    model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
    if "SIMPLE" in model:           # f, cx, cy, ...
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]   # PINHOLE / OPENCV: fx, fy, cx, cy


def _fit_depth_model(pred: np.ndarray, metric: np.ndarray) -> Tuple[str, np.ndarray, float]:
    """Fit pred->metric with a direct and a disparity model; return the better.

    direct:     metric ≈ a*pred + b
    disparity:  1/metric ≈ a*pred + b   (mono nets are often disparity-like)
    """
    best = None
    # direct
    A = np.stack([pred, np.ones_like(pred)], 1)
    coef, *_ = np.linalg.lstsq(A, metric, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ coef - metric) ** 2)))
    best = ("direct", coef, rmse)
    # disparity (guard against zero/negative metric)
    if np.all(metric > 1e-6):
        inv = 1.0 / metric
        coef2, *_ = np.linalg.lstsq(A, inv, rcond=None)
        pred_inv = A @ coef2
        ok = pred_inv > 1e-6
        if ok.sum() > len(pred) * 0.5:
            rmse2 = float(np.sqrt(np.mean((1.0 / pred_inv[ok] - metric[ok]) ** 2)))
            if rmse2 < rmse:
                best = ("disparity", coef2, rmse2)
    return best


def _apply_model(model: str, coef: np.ndarray, pred_map: np.ndarray) -> np.ndarray:
    lin = coef[0] * pred_map + coef[1]
    if model == "direct":
        return lin
    with np.errstate(divide="ignore"):
        out = 1.0 / lin
    out[lin <= 1e-6] = np.nan
    return out


def _voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float,
                      max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    xyz, rgb = xyz[idx], rgb[idx]
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    return xyz, rgb


def build_dense_seed_cloud(sparse_dir: Path, frames_dir: Path,
                           depth_estimator, out_ply: Path,
                           pixel_stride: int = 3, min_anchors: int = 25,
                           max_points: int = 600_000,
                           voxel: float = 0.0) -> Optional[Path]:
    """Generate the dense seed cloud. Returns the .ply path, or None on failure."""
    import cv2
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    used, skipped = 0, 0

    # Auto voxel size from a ROBUST scene extent (COLMAP has far outlier points
    # that would otherwise inflate the bounding box and over-decimate the cloud).
    if voxel <= 0:
        pts = np.array([p.xyz for p in rec.points3D.values()])
        if len(pts):
            lo, hi = np.percentile(pts, [5, 95], axis=0)
            extent = float(np.linalg.norm(hi - lo))
        else:
            extent = 1.0
        voxel = max(extent / 400.0, 1e-4)
    log.info("Dense seed: scene voxel=%.5f", voxel)

    for im in rec.images.values():
        img_path = frames_dir / im.name
        if not img_path.exists():
            skipped += 1
            continue
        rgb_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        H, W = rgb_img.shape[:2]
        pred = depth_estimator.estimate(rgb_img)
        if pred.shape != (H, W):
            pred = cv2.resize(pred, (W, H))

        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = _intrinsics(cam)
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = cfw.rotation.matrix()
        t = np.asarray(cfw.translation, float)

        # anchors: visible 3D points -> (pixel, metric depth)
        px, py, zc = [], [], []
        for p2d in im.points2D:
            has = p2d.has_point3D() if hasattr(p2d, "has_point3D") else (p2d.point3D_id != -1)
            if not has:
                continue
            X = np.asarray(rec.points3D[p2d.point3D_id].xyz, float)
            z = float((R @ X + t)[2])
            if z <= 0:
                continue
            u, v = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
            if 0 <= u < W and 0 <= v < H:
                px.append(u); py.append(v); zc.append(z)
        if len(zc) < min_anchors:
            skipped += 1
            continue

        a_pred = pred[np.array(py), np.array(px)]
        a_metric = np.array(zc)
        # Trim anchor outliers (COLMAP triangulates far junk on textureless walls)
        # to a robust depth band before fitting.
        q_lo, q_hi = np.percentile(a_metric, [10, 90])
        inl = (a_metric >= q_lo) & (a_metric <= q_hi)
        if inl.sum() < min_anchors:
            inl = np.ones_like(a_metric, bool)
        model, coef, rmse = _fit_depth_model(a_pred[inl], a_metric[inl])
        med = float(np.median(a_metric[inl]))
        if rmse > 0.5 * med:          # alignment too poor -> skip this frame
            skipped += 1
            continue

        metric = _apply_model(model, coef, pred)
        lo, hi = np.percentile(a_metric[inl], [5, 95])
        lo, hi = 0.7 * lo, 1.3 * hi
        # Sanity: the densified depth median must track the anchor median.
        mvalid = np.isfinite(metric) & (metric > lo) & (metric < hi)
        if not mvalid.any() or abs(np.median(metric[mvalid]) - med) > 1.5 * med:
            skipped += 1
            continue

        ys, xs = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
        z = metric[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(z) & (z > lo) & (z < hi)
        zc2 = z[valid]
        xc = (xs[valid] - cx) / fx * zc2
        yc = (ys[valid] - cy) / fy * zc2
        pts_cam = np.stack([xc, yc, zc2], 1)
        pts_world = (R.T @ (pts_cam - t).T).T
        cols = rgb_img[::pixel_stride, ::pixel_stride][valid]
        all_xyz.append(pts_world); all_rgb.append(cols)
        used += 1

    if not all_xyz:
        log.warning("Dense seed: no frames produced points (poor alignment).")
        return None

    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    # Clip far outliers to a robust bounding box (1st-99th pct, padded).
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    pad = 0.25 * (hi - lo)
    keep = np.all((xyz >= lo - pad) & (xyz <= hi + pad), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    xyz, rgb = _voxel_downsample(xyz, rgb, voxel, max_points)
    log.info("Dense seed: %d frames used, %d skipped -> %d points (voxel %.4f)",
             used, skipped, len(xyz), voxel)

    from ..splat.pointcloud import write_ply
    return write_ply(xyz, rgb, out_ply)
