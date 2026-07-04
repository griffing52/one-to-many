"""Render a coloured point cloud from an arbitrary camera (z-buffer splat).

The novel-view backbone for the metric scene clouds: project the base-frame
points through a :class:`o2m.splat.camera.Camera`, keep the nearest point per
pixel (painter's order, optional 2x2 kernel to close pin-holes), then inpaint
the remaining holes. Returns depth alongside RGB so callers can depth-composite
the URDF robot into the same view.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def render_points(xyz: np.ndarray, rgb: np.ndarray, camera,
                  kernel: int = 2, fill_method: str = "inpaint",
                  max_hole_frac: float = 1.0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render (rgb HxWx3 uint8, depth HxW float32 [0 = empty], covered HxW bool).

    Args:
        xyz: (N,3) world/base-frame points.
        rgb: (N,3) uint8 colours.
        camera: Camera with intrinsics + OpenCV c2w.
        kernel: splat each point to a kernel x kernel pixel block (1 = single).
        fill_method: hole fill for uncovered pixels (see wrist_warp.FILL_METHODS);
            holes are filled in RGB only — depth stays 0 there ("unknown").
        max_hole_frac: skip filling if more than this fraction is empty
            (a mostly-empty view is better shown honestly than smeared).
    """
    H, W = camera.height, camera.width
    w2c = np.linalg.inv(camera.c2w)
    p = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = p[:, 2]
    ok = z > 1e-3
    u = p[ok, 0] / z[ok] * camera.fx + camera.cx
    v = p[ok, 1] / z[ok] * camera.fy + camera.cy
    z = z[ok]
    cols = rgb[ok]
    inb = (u > -1) & (u < W) & (v > -1) & (v < H)
    u, v, z, cols = u[inb], v[inb], z[inb], cols[inb]

    out = np.zeros((H, W, 3), np.uint8)
    depth = np.zeros((H, W), np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    order = np.argsort(-z)                     # far -> near, near wins
    u, v, z, cols = u[order], v[order], z[order], cols[order]
    offs = [(du, dv) for du in range(kernel) for dv in range(kernel)]
    for du, dv in offs:
        iu = np.floor(u + du).astype(int)
        iv = np.floor(v + dv).astype(int)
        m = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
        iu, iv, zz, cc = iu[m], iv[m], z[m], cols[m]
        closer = zz < zbuf[iv, iu]
        iu, iv, zz, cc = iu[closer], iv[closer], zz[closer], cc[closer]
        zbuf[iv, iu] = zz
        out[iv, iu] = cc
        depth[iv, iu] = zz

    covered = depth > 0
    hole_frac = 1.0 - covered.mean()
    if fill_method != "none" and 0 < hole_frac <= max_hole_frac:
        from ..worldmodel.wrist_warp import fill_holes
        out = fill_holes(out, covered, fill_method)
    return out, depth, covered
