"""Convert a COLMAP sparse model to a Nerfstudio ``transforms.json``.

See docs/data_contracts.md for the authoritative schema. Key conversions:

1. COLMAP gives world->cam (R_wc, t_wc) in OpenCV convention.
2. cam->world: R_cw = R_wc.T,  C = -R_wc.T @ t_wc.
3. Nerfstudio expects OpenGL camera axes -> flip local y,z (CV_TO_GL).

We keep raw poses (no recentre/rescale) and disable nerfstudio auto-orient so
the splat world frame == COLMAP frame, which the robot-base sim3 alignment
relies on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils import geometry as geom
from .model_io import ColmapModel, qvec_to_rotmat


def _intrinsics_dict(cam) -> Dict:
    """Map a COLMAP camera's params into transforms.json intrinsic fields."""
    p = cam.params
    out = {"w": cam.width, "h": cam.height}
    model = cam.model
    if model == "SIMPLE_PINHOLE":          # f, cx, cy
        out.update(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2])
    elif model == "PINHOLE":               # fx, fy, cx, cy
        out.update(fl_x=p[0], fl_y=p[1], cx=p[2], cy=p[3])
    elif model == "SIMPLE_RADIAL":         # f, cx, cy, k1
        out.update(fl_x=p[0], fl_y=p[0], cx=p[1], cy=p[2], k1=p[3])
    elif model == "OPENCV":                # fx, fy, cx, cy, k1, k2, p1, p2
        out.update(fl_x=p[0], fl_y=p[1], cx=p[2], cy=p[3],
                   k1=p[4], k2=p[5], p1=p[6], p2=p[7])
    else:
        out.update(fl_x=p[0], fl_y=p[1] if len(p) > 1 else p[0],
                   cx=cam.width / 2, cy=cam.height / 2)
    out["camera_model"] = "OPENCV"
    return out


def colmap_to_transforms(model: ColmapModel,
                         image_subdir: str = "images",
                         mask_subdir: Optional[str] = "masks",
                         ply_file_path: Optional[str] = None,
                         name_prefix: Optional[str] = None) -> Dict:
    """Build the transforms.json dict from an in-memory COLMAP model.

    Args:
        name_prefix: if set, keep only frames whose image name starts with it
            (e.g. ``"wrist/"`` to drop the ZED frames from a combined model).
    """
    selected = [im for im in model.images.values()
                if name_prefix is None or im.name.startswith(name_prefix)]
    # Intrinsics from the selected frames' camera (single shared camera).
    cam_id = selected[0].camera_id if selected else next(iter(model.cameras))
    out: Dict = _intrinsics_dict(model.cameras[cam_id])

    frames: List[Dict] = []
    for img in sorted(selected, key=lambda im: im.name):
        R_wc = qvec_to_rotmat(img.qvec)
        t_wc = img.tvec
        R_cw = R_wc.T
        C = -R_wc.T @ t_wc
        c2w_cv = geom.make_se3(R_cw, C)
        c2w_gl = geom.opencv_c2w_to_opengl(c2w_cv)

        frame = {
            "file_path": f"{image_subdir}/{img.name}",
            "transform_matrix": c2w_gl.tolist(),
        }
        if mask_subdir is not None:
            frame["mask_path"] = f"{mask_subdir}/{img.name}.png"
        frames.append(frame)

    out["frames"] = frames
    if ply_file_path is not None:
        out["ply_file_path"] = ply_file_path
    return out


def write_transforms(transforms: Dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(transforms, f, indent=2)
    return path
