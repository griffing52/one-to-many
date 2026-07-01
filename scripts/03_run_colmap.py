#!/usr/bin/env python3
"""Stage 03 — recover camera poses with COLMAP and write transforms.json.

Runs feature -> match -> map on the masked wrist frames (single shared camera),
reads the sparse model, exports a seed point cloud, and converts poses to a
Nerfstudio transforms.json (raw poses; no recentre/rescale).
"""
from __future__ import annotations

import shutil

import numpy as np

from _common import base_parser, load, log

from o2m.colmap import ColmapRunner, read_model
from o2m.colmap.to_nerfstudio import colmap_to_transforms, write_transforms


def _write_ply(xyz: np.ndarray, rgb: np.ndarray, path) -> None:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb.astype(int)):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg, paths = load(args)

    model_str = cfg.get("camera.colmap_model", "OPENCV")
    w = cfg.get("camera.wrist", {})
    cam_params = None
    if all(k in w for k in ("fx_prior", "fy_prior", "cx_prior", "cy_prior")):
        cam_params = f"{w['fx_prior']},{w['fy_prior']},{w['cx_prior']},{w['cy_prior']},0,0,0,0"

    runner = ColmapRunner(
        binary=cfg.get("colmap.binary", "colmap"),
        camera_model=model_str,
        single_camera=bool(cfg.get("colmap.single_camera", True)),
        matcher=cfg.get("colmap.matcher", "exhaustive"),
        camera_params=cam_params,
        backend=cfg.get("colmap.backend_impl", "auto"),
        sift_options=cfg.get("colmap.sift", None),
        mapper_options=cfg.get("colmap.mapper_tuning", None),
    )
    if not runner.available():
        raise SystemExit(
            "No COLMAP backend found. Install `pycolmap` (pip/uv) or the "
            "`colmap` CLI binary."
        )

    mask_dir = paths.masks if cfg.get("colmap.use_masks", True) else None
    sparse0 = runner.run(paths.frames, mask_dir, paths.colmap_db, paths.colmap / "sparse")

    model = read_model(sparse0)
    log.info("COLMAP: %d images registered, %d sparse points",
             len(model.images), len(model.points_xyz))

    # Only carry masks into training if masking is actually enabled; otherwise
    # transforms.json must NOT reference mask_path (else stale/bad masks are used).
    use_masks = bool(cfg.get("colmap.use_masks", False))

    # Seed point cloud for splatfacto + symlinked images (+masks) for nerfstudio.
    paths.nerfstudio.mkdir(parents=True, exist_ok=True)
    ply_rel = "sparse_pc.ply"
    if len(model.points_xyz):
        _write_ply(model.points_xyz, model.points_rgb, paths.nerfstudio / ply_rel)

    links = [("images", paths.frames)] + ([("masks", paths.masks)] if use_masks else [])
    for sub, src in links:
        link = paths.nerfstudio / sub
        if link.exists() or link.is_symlink():
            link.unlink()
        try:
            link.symlink_to(src.resolve())
        except OSError:
            shutil.copytree(src, link)

    transforms = colmap_to_transforms(
        model, image_subdir="images", mask_subdir="masks" if use_masks else None,
        ply_file_path=ply_rel if len(model.points_xyz) else None,
    )
    write_transforms(transforms, paths.transforms_json)
    log.info("Wrote %s (%d frames)", paths.transforms_json, len(transforms["frames"]))


if __name__ == "__main__":
    main()
