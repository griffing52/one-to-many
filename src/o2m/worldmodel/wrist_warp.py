"""Detail-preserving wrist (RealSense) view synthesis by depth-warping.

Instead of rendering the low-texture splat for the wrist view (streaky), we take
the REAL recorded wrist frame, lift it to 3D with monocular depth, move the
virtual camera by the per-frame offset, and reproject. This keeps the full real
detail of the shelf / handles / bag and is exact for small translations.

Two robustness pieces, both tunable in ``configs/worldmodel.yaml``:

- **Gripper kept fixed**: the gripper is rigidly mounted to the wrist camera, so
  it must NOT move when the scene shifts. Pixels inside a bottom-centre trapezoid
  are copied straight from the original frame. (The trapezoid width is halved vs
  the first version, per request — see ``gripper_mask`` in the config.)
- **Kernel-splat + inpaint**: each source pixel is splatted to a 2x2 block to
  close pin-holes, and remaining disocclusion holes are inpainted.

The camera-frame offset is derived from the base-frame EE offset via forward
kinematics of the ``hand_cam`` link, remapped from the URDF link axes (z-forward,
x-up) to the optical convention (x-right, y-down, z-forward) by ``OPTICAL_FROM_LINK``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# hand_cam URDF link axes (z fwd, x up, y right-ish) -> OpenCV optical
# (x right, y down, z fwd): optical_x = link_y, optical_y = -link_x, optical_z = link_z.
OPTICAL_FROM_LINK = np.array([[0.0, -1.0, 0.0],
                              [1.0,  0.0, 0.0],
                              [0.0,  0.0, 1.0]])


@dataclass
class WristIntrinsics:
    fx: float = 465.0
    fy: float = 465.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480


@dataclass
class GripperMask:
    """Bottom-centre trapezoid kept fixed (not warped). Fractions of image size."""
    y_start_frac: float = 0.70   # trapezoid top, as a fraction down the image
    half_top: float = 0.05       # half-width at y_start (fraction of W)   [was 0.10]
    half_grow: float = 0.175     # extra half-width at the bottom edge     [was 0.35]

    def mask(self, h: int, w: int) -> np.ndarray:
        ys, xs = np.mgrid[0:h, 0:w]
        y0 = self.y_start_frac * h
        frac = np.clip((ys - y0) / max(1.0, (h - y0)), 0.0, 1.0)
        half = (self.half_top + self.half_grow * frac) * w
        return (ys > y0) & (np.abs(xs - 0.5 * w) < half)


def disparity_to_depth(disp: np.ndarray) -> np.ndarray:
    """Depth-Anything relative disparity -> a positive, median-normalised depth."""
    d = 1.0 / (disp - disp.min() + 0.3)
    return d * (0.5 / np.median(d))


def disparities_to_depths(disps: np.ndarray) -> np.ndarray:
    """Video-Depth-Anything disparity stack (N,H,W) -> depths with ONE global
    shift/scale. The stack is already temporally consistent; normalising each
    frame separately (as :func:`disparity_to_depth` does) would reintroduce the
    per-frame scale flicker the video model removes."""
    d = 1.0 / (disps - disps.min() + 0.3)
    return d * (0.5 / np.median(d))


# Hole-fill strategies for the disocclusion gaps left by forward warping. All take
# the scattered image and a boolean ``filled`` mask (True where a pixel got a value)
# and return the completed image. Pick via ``WristWarper(fill_method=...)`` or
# compare them all with ``scripts/09_fill_methods_demo.py``.
FILL_METHODS = ("none", "nearest", "bilinear", "edge_aware", "inpaint", "telea", "ns")


def fill_holes(img: np.ndarray, filled: np.ndarray, method: str = "inpaint") -> np.ndarray:
    """Fill ``~filled`` pixels of ``img`` using ``method`` (see :data:`FILL_METHODS`)."""
    holes = ~filled
    if method == "none" or not holes.any():
        return img
    if method == "nearest":
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(holes, return_distances=False,
                                             return_indices=True)
        return img[tuple(idx)]
    if method == "bilinear":
        from scipy.interpolate import griddata
        ys, xs = np.nonzero(filled)
        pts = np.stack([ys, xs], 1)
        qy, qx = np.nonzero(holes)
        out = img.copy()
        vals = griddata(pts, img[ys, xs], (qy, qx), method="linear")
        nn = griddata(pts, img[ys, xs], (qy, qx), method="nearest")  # convex-hull gaps
        vals = np.where(np.isnan(vals), nn, vals)
        out[qy, qx] = np.clip(vals, 0, 255).astype(img.dtype)
        return out
    # OpenCV inpainting variants
    import cv2
    mask = holes.astype(np.uint8) * 255
    if method in ("edge_aware", "ns"):
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)       # Navier-Stokes (edge-aware)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)        # fast-marching ("small inpaint")


def base_offset_to_camera(delta_world: np.ndarray, cam_R_base: np.ndarray) -> np.ndarray:
    """Base-frame EE offset -> optical camera-frame offset (x right, y down, z fwd).

    ``cam_R_base`` is the ``hand_cam`` link rotation (base <- link) from FK.
    """
    return OPTICAL_FROM_LINK.T @ (cam_R_base.T @ np.asarray(delta_world, float))


class WristWarper:
    def __init__(self, intr: WristIntrinsics, gripper_mask: GripperMask,
                 kernel_splat: bool = True, inpaint_holes: bool = True,
                 fill_method: str = "inpaint"):
        self.intr = intr
        self.gmask = gripper_mask
        self.kernel_splat = kernel_splat
        # Back-compat: inpaint_holes=False forces no fill regardless of fill_method.
        self.fill_method = fill_method if inpaint_holes else "none"

    def scatter(self, real_rgb: np.ndarray, depth: np.ndarray, dcam: np.ndarray,
                gmask: np.ndarray = None):
        """Forward-warp only: returns (scattered rgb, filled bool mask). No fill/gripper."""
        return self._scatter(real_rgb, depth, dcam, exclude=gmask)

    def warp(self, real_rgb: np.ndarray, depth: np.ndarray,
             dcam: np.ndarray, fill_method: str = None,
             gmask: np.ndarray = None) -> np.ndarray:
        """Warp ``real_rgb`` as if the camera moved by ``dcam`` (optical frame).

        Args:
            real_rgb: HxWx3 uint8 original wrist frame.
            depth: HxW positive depth (from :func:`disparity_to_depth`).
            dcam: (3,) optical-frame camera translation (m).
            fill_method: override the instance hole-fill strategy for this call.
            gmask: per-frame gripper mask (e.g. TemporalGripperMasker) overriding
                the static trapezoid. Masked pixels are EXCLUDED from the warp
                (they are camera-rigid; warping them at near depth sprays) and
                pasted back from the original frame.
        """
        gm = gmask if gmask is not None else self.gmask.mask(*depth.shape)
        out, filled = self._scatter(real_rgb, depth, dcam, exclude=gm)
        out = fill_holes(out, filled | gm, fill_method or self.fill_method)
        # Keep the gripper exactly where it was in the original frame.
        out[gm] = real_rgb[gm]
        return out

    def _scatter(self, real_rgb: np.ndarray, depth: np.ndarray, dcam: np.ndarray,
                 exclude: np.ndarray = None):
        intr = self.intr
        h, w = depth.shape
        ys, xs = np.mgrid[0:h, 0:w]
        z = depth
        xc = (xs - intr.cx) / intr.fx * z
        yc = (ys - intr.cy) / intr.fy * z
        # Move the virtual camera by +dcam => points shift by -dcam in its frame.
        xc2, yc2, zc2 = xc - dcam[0], yc - dcam[1], z - dcam[2]
        valid = zc2 > 1e-3
        if exclude is not None:
            valid &= ~exclude
        u = (xc2 / np.where(valid, zc2, 1.0) * intr.fx + intr.cx)
        v = (yc2 / np.where(valid, zc2, 1.0) * intr.fy + intr.cy)

        out = np.zeros_like(real_rgb)
        filled = np.zeros((h, w), bool)
        zbuf = np.full((h, w), np.inf)
        src_rgb = real_rgb.reshape(-1, 3)
        # Painter's order: far -> near so nearer points win the z-buffer.
        order = np.argsort(-zc2.ravel())
        uu = u.ravel()[order]
        vv = v.ravel()[order]
        zz = zc2.ravel()[order]
        vv_ok = valid.ravel()[order]
        cc = src_rgb[order]
        offs = [(0, 0), (1, 0), (0, 1), (1, 1)] if self.kernel_splat else [(0, 0)]
        for du, dv in offs:
            iu = np.floor(uu + du).astype(int)
            iv = np.floor(vv + dv).astype(int)
            m = vv_ok & (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
            iu, iv, zc, col = iu[m], iv[m], zz[m], cc[m]
            closer = zc < zbuf[iv, iu]
            iu, iv, col = iu[closer], iv[closer], col[closer]
            zbuf[iv, iu] = zc[closer]
            out[iv, iu] = col
            filled[iv, iu] = True
        return out, filled
