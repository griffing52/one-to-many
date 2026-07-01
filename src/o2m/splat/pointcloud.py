"""Export point clouds and depth from a trained splat.

Two representation-agnostic outputs that are far more robust than 3DGS
novel-view rendering in low-texture / sparse-viewpoint scenes:

- ``export_gaussian_pointcloud``: the Gaussian centres + colours as a coloured
  .ply, read directly from the checkpoint (no pymeshlab / ns-export needed).
- ``render_depth``: a metric-ish depth map from any camera (the splat's depth
  channel), plus a colourised PNG for inspection.
- ``unproject_depth_to_points``: back-project a rendered depth map into a dense
  coloured point cloud for a given view.

These give downstream consumers a geometry handle (points/depth) that does not
depend on the splat looking good from novel angles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Spherical-harmonic DC term -> RGB (gsplat/3DGS convention).
_SH_C0 = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gaussian_points_from_model(splat_model, opacity_threshold: float = 0.1
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xyz [N,3], rgb uint8 [N,3]) for the visible Gaussian centres."""
    gp = splat_model._pipeline.model.gauss_params
    means = gp["means"].detach().cpu().numpy()
    dc = gp["features_dc"].detach().cpu().numpy()
    opac = _sigmoid(gp["opacities"].detach().cpu().numpy().reshape(-1))

    rgb = np.clip(_SH_C0 * dc + 0.5, 0.0, 1.0)
    keep = opac >= opacity_threshold
    return means[keep], (rgb[keep] * 255).astype(np.uint8)


def write_ply(xyz: np.ndarray, rgb: np.ndarray, path: Path) -> Path:
    """Write a coloured ASCII .ply (portable, no external deps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb.astype(int)):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    return path


def export_gaussian_pointcloud(splat_model, path: Path,
                               opacity_threshold: float = 0.1) -> Path:
    xyz, rgb = gaussian_points_from_model(splat_model, opacity_threshold)
    return write_ply(xyz, rgb, path)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map a depth array to an 8-bit colour image for inspection."""
    import cv2

    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)[..., ::-1]  # BGR->RGB


def unproject_depth_to_points(camera, depth: np.ndarray, rgb: np.ndarray,
                              stride: int = 2, max_depth: Optional[float] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map into world-frame coloured points.

    Uses the camera intrinsics + OpenCV c2w. Returns (xyz [M,3], rgb [M,3]).
    """
    H, W = depth.shape[:2]
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride].astype(np.float32)
    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        valid &= z < max_depth

    x = (xs - camera.cx) / camera.fx * z
    y = (ys - camera.cy) / camera.fy * z
    pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=1)  # OpenCV cam frame
    pts_world = (camera.c2w[:3, :3] @ pts_cam.T).T + camera.c2w[:3, 3]
    cols = rgb[::stride, ::stride][valid]
    return pts_world, cols
