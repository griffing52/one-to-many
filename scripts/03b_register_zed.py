#!/usr/bin/env python3
"""Stage 03b (optional) — register the stationary ZED camera into the splat frame.

Runs ONE combined COLMAP reconstruction over the wrist frames + a few ZED frames
(PER_FOLDER -> one camera each), so the ZED pose ends up in the same world frame
as the wrist poses used to train the splat.

This SUPERSEDES stage 03: it rewrites transforms.json (wrist frames only, in the
combined frame) and additionally writes align/zed_camera.json. Run 03b instead
of 03 when you want robot_overlay rendered from the third-person demo view.

Graceful degradation: if the ZED does not register (its far third-person view
may share too few features with the wrist close-ups), the wrist-only transforms
are still written and rendering falls back to a wrist viewpoint.
"""
from __future__ import annotations

import shutil

import numpy as np

from _common import base_parser, load, log

from o2m.align import extract_zed_camera, write_zed_camera
from o2m.colmap import ColmapRunner, read_model
from o2m.colmap.to_nerfstudio import colmap_to_transforms, write_transforms
from o2m.data import Episode


def _write_ply(xyz, rgb, path):
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
    p.add_argument("--num-zed", type=int, default=8, help="ZED frames to include.")
    args = p.parse_args()
    cfg, paths = load(args)

    # Build a combined image dir: combined/wrist/* and combined/zed/*.
    combined = paths.colmap / "combined"
    if combined.exists():
        shutil.rmtree(combined)
    (combined / "wrist").mkdir(parents=True)
    (combined / "zed").mkdir(parents=True)

    for src in sorted(paths.frames.glob("*.png")):
        shutil.copy2(src, combined / "wrist" / src.name)

    ep = Episode(paths.raw_episode,
                 wrist_dir=cfg.get("dataset.cameras.wrist", "realsense_color"),
                 zed_dir=cfg.get("dataset.cameras.zed", "zed_color"))
    zed_frames = ep.zed_frames()
    pick = np.linspace(0, len(zed_frames) - 1, args.num_zed).astype(int)
    for i in pick:
        shutil.copy2(zed_frames[i], combined / "zed" / zed_frames[i].name)
    log.info("Combined set: %d wrist + %d zed frames",
             len(list((combined / 'wrist').glob('*.png'))), len(pick))

    runner = ColmapRunner(
        camera_model=cfg.get("camera.colmap_model", "OPENCV"),
        single_camera=False, camera_mode="per_folder",
        matcher=cfg.get("colmap.matcher", "exhaustive"),
        camera_params=None,           # two different cameras -> let COLMAP estimate
        backend=cfg.get("colmap.backend_impl", "auto"),
        sift_options=cfg.get("colmap.sift", None),
        mapper_options=cfg.get("colmap.mapper_tuning", None),
    )
    if not runner.available():
        raise SystemExit("No COLMAP backend (install pycolmap or the colmap CLI).")

    sparse0 = runner.run(combined, None, paths.colmap / "db_combined.db",
                         paths.colmap / "sparse_combined")
    model = read_model(sparse0)
    n_wrist = sum(im.name.startswith("wrist/") for im in model.images.values())
    n_zed = sum(im.name.startswith("zed/") for im in model.images.values())
    log.info("Combined model: %d wrist + %d zed registered, %d points",
             n_wrist, n_zed, len(model.points_xyz))

    # transforms.json: wrist frames only, in the combined frame.
    paths.nerfstudio.mkdir(parents=True, exist_ok=True)
    ply_rel = "sparse_pc.ply"
    if len(model.points_xyz):
        _write_ply(model.points_xyz, model.points_rgb, paths.nerfstudio / ply_rel)

    link = paths.nerfstudio / "images"
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(combined.resolve())
    except OSError:
        shutil.copytree(combined, link)

    transforms = colmap_to_transforms(
        model, image_subdir="images", mask_subdir=None,
        ply_file_path=ply_rel if len(model.points_xyz) else None,
        name_prefix="wrist/",
    )
    write_transforms(transforms, paths.transforms_json)
    log.info("Wrote %s (%d wrist frames)", paths.transforms_json, len(transforms["frames"]))

    # ZED camera (the demo viewpoint), if it registered.
    zed_cam = extract_zed_camera(model, zed_prefix="zed")
    if zed_cam is None:
        log.warning("ZED did not register -> no zed_camera.json. robot_overlay will "
                    "fall back to a wrist viewpoint. (Scene texture / viewpoint gap.)")
    else:
        write_zed_camera(zed_cam, paths.align / "zed_camera.json")
        log.info("Wrote ZED viewpoint -> %s (from %s)",
                 paths.align / "zed_camera.json", zed_cam["source_image"])


if __name__ == "__main__":
    main()
