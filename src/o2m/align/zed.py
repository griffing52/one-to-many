"""Register the stationary ZED camera into the splat frame.

Strategy: run ONE COLMAP reconstruction containing both the moving wrist frames
and a few stationary ZED frames (``PER_FOLDER`` -> one camera per folder). The
ZED pose then comes out in the *same* world frame as the wrist poses used to
train the splat, so no cross-model alignment is needed.

This is an optional layer: if the ZED does not register (its third-person view
may share too few features with the wrist close-ups), the wrist-only transforms
are still written and rendering falls back to a wrist viewpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..colmap.model_io import ColmapModel, qvec_to_rotmat
from ..utils import geometry as geom


def split_model_by_folder(model: ColmapModel, prefix: str):
    """Return image objects whose COLMAP name starts with ``<prefix>/``."""
    return [im for im in model.images.values() if im.name.startswith(f"{prefix}/")]


def camera_json_from_image(model: ColmapModel, image) -> Dict:
    """Build a Camera-style json dict (intrinsics + OpenCV c2w) for one image."""
    cam = model.cameras[image.camera_id]
    p = cam.params
    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:  # SIMPLE_* : single focal
        fx = fy = p[0]
        cx, cy = p[1], p[2]

    R_wc = qvec_to_rotmat(image.qvec)
    C = -R_wc.T @ image.tvec
    c2w = geom.make_se3(R_wc.T, C)  # OpenCV cam->world in the splat frame
    return {
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "width": int(cam.width), "height": int(cam.height),
        "c2w": c2w.tolist(),
        "source_image": image.name,
    }


def extract_zed_camera(model: ColmapModel, zed_prefix: str = "zed") -> Optional[Dict]:
    """Pick the best-registered ZED image and return its camera json, or None."""
    zed_images = split_model_by_folder(model, zed_prefix)
    if not zed_images:
        return None
    # The ZED is static; any registered frame works. Pick deterministically by name.
    best = sorted(zed_images, key=lambda im: im.name)[0]
    return camera_json_from_image(model, best)


def write_zed_camera(cam_json: Dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cam_json, indent=2))
    return path
