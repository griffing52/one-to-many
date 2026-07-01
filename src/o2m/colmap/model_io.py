"""Read a COLMAP sparse model.

Uses ``pycolmap`` when available; otherwise falls back to the standard binary
parsers for ``cameras.bin`` / ``images.bin`` / ``points3D.bin``. Returns the
per-image world->camera pose (OpenCV convention) and intrinsics, plus the
sparse seed point cloud used to initialise splatfacto.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class ColmapCamera:
    model: str
    width: int
    height: int
    params: np.ndarray  # model-dependent intrinsics


@dataclass
class ColmapImage:
    name: str
    qvec: np.ndarray    # (4,) world->cam quaternion (w, x, y, z)
    tvec: np.ndarray    # (3,) world->cam translation
    camera_id: int


@dataclass
class ColmapModel:
    cameras: Dict[int, ColmapCamera]
    images: Dict[int, ColmapImage]
    points_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    points_rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# --- binary readers (fallback when pycolmap is absent) ---------------------

def _read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack(fmt, data)


def _read_cameras_bin(path: Path) -> Dict[int, ColmapCamera]:
    # COLMAP camera model id -> name (subset we care about).
    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                   3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE"}
    num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
    cameras: Dict[int, ColmapCamera] = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            cam_id, model_id, w, h = _read_next_bytes(f, 24, "iiQQ")
            k = num_params[model_id]
            params = np.array(_read_next_bytes(f, 8 * k, "d" * k))
            cameras[cam_id] = ColmapCamera(model_names[model_id], int(w), int(h), params)
    return cameras


def _read_images_bin(path: Path) -> Dict[int, ColmapImage]:
    images: Dict[int, ColmapImage] = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            img_id = _read_next_bytes(f, 4, "i")[0]
            qvec = np.array(_read_next_bytes(f, 32, "dddd"))
            tvec = np.array(_read_next_bytes(f, 24, "ddd"))
            cam_id = _read_next_bytes(f, 4, "i")[0]
            name = ""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c.decode("utf-8")
            num_pts = _read_next_bytes(f, 8, "Q")[0]
            f.read(24 * num_pts)  # skip 2D points (x, y, point3D_id)
            images[img_id] = ColmapImage(name, qvec, tvec, cam_id)
    return images


def _read_points3d_bin(path: Path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            _read_next_bytes(f, 8, "Q")  # point id
            xyz.append(_read_next_bytes(f, 24, "ddd"))
            rgb.append(_read_next_bytes(f, 3, "BBB"))
            _read_next_bytes(f, 8, "d")  # reproj error
            track_len = _read_next_bytes(f, 8, "Q")[0]
            f.read(8 * track_len)
    return np.array(xyz).reshape(-1, 3), np.array(rgb).reshape(-1, 3)


def read_model(sparse_dir: Path) -> ColmapModel:
    sparse_dir = Path(sparse_dir)
    try:
        import pycolmap  # type: ignore

        rec = pycolmap.Reconstruction(str(sparse_dir))
        cameras = {
            cid: ColmapCamera(cam.model_name, cam.width, cam.height, np.array(cam.params))
            for cid, cam in rec.cameras.items()
        }
        images = {
            iid: ColmapImage(im.name, np.array(im.cam_from_world.rotation.quat)[[3, 0, 1, 2]]
                             if hasattr(im, "cam_from_world") else np.array(im.qvec),
                             np.array(im.cam_from_world.translation)
                             if hasattr(im, "cam_from_world") else np.array(im.tvec),
                             im.camera_id)
            for iid, im in rec.images.items()
        }
        pts = np.array([p.xyz for p in rec.points3D.values()]) if rec.points3D else np.zeros((0, 3))
        rgb = np.array([p.color for p in rec.points3D.values()]) if rec.points3D else np.zeros((0, 3))
        return ColmapModel(cameras, images, pts, rgb)
    except Exception:
        cameras = _read_cameras_bin(sparse_dir / "cameras.bin")
        images = _read_images_bin(sparse_dir / "images.bin")
        pts, rgb = _read_points3d_bin(sparse_dir / "points3D.bin")
        return ColmapModel(cameras, images, pts, rgb)
