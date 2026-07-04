#!/usr/bin/env python3
"""Stage 10 — metric ZED scene depth + base-frame scene point cloud.

Fits the metric scale of (Video-)Depth-Anything on the ZED view using the
FK-tracked green gripper (see ``o2m.worldmodel.scene_cloud``), applies it to the
robot-free CLEAN PLATE, and unprojects through the calibrated extrinsic into a
base-frame scene cloud.

    PYTHONPATH=src python scripts/10_zed_metric_scene.py

Outputs:
  - outputs/<ep>/align/zed_scene_metric.npz   (depth, K, c2w, a, b, diag json)
  - outputs/<ep>/pointcloud/zed_scene_metric.ply
  - outputs/<ep>/renders/worldmodel/zed_scene_depth.png (turbo colormap check)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from o2m.config import Config  # noqa: E402
from o2m.data import Episode, load_joint_trajectory  # noqa: E402
from o2m.robot import PiperModel  # noqa: E402
from o2m.utils import get_logger  # noqa: E402
from o2m.worldmodel.scene_cloud import build_zed_metric_scene  # noqa: E402

log = get_logger("o2m.scripts.zed_scene")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worldmodel.yaml")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--stride", type=int, default=8, help="ZED frame sampling stride.")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--max-depth", type=float, default=3.5,
                    help="Clip the scene cloud beyond this depth (m).")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    wm = cfg.section("worldmodel")
    root = Path(cfg.source).resolve().parent.parent

    def _abs(p):
        q = Path(p)
        return q if q.is_absolute() else (root / q).resolve()

    episode = args.episode or wm["episode"]
    ep = Episode(_abs(wm["data_root"]) / episode)
    joints = load_joint_trajectory(ep.actions_df(), arm=cfg.get("dataset.arm", "slave"))

    from PIL import Image
    plate = np.asarray(Image.open(_abs(wm["clean_plate"])).convert("RGB"))
    ext = np.load(str(_abs(wm["zed_extrinsic_npz"])))
    K, c2w = ext["K"], ext["c2w"]

    model = PiperModel(cfg.require("robot.urdf"), cfg.require("robot.urdf_dir"),
                       base_frame=cfg.get("robot.base_frame", "base_link"),
                       ee_frame=cfg.get("robot.ee_frame"),
                       camera_frame=cfg.get("robot.camera_frame") or "hand_cam")

    depth, fit, diag = build_zed_metric_scene(
        ep.zed_frames(), joints, plate, K, c2w, model,
        stride=args.stride, encoder=args.encoder)

    out_root = root / "outputs" / episode
    npz_path = out_root / "align" / "zed_scene_metric.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, depth=depth, K=K, c2w=c2w, a=fit.a, b=fit.b,
             diag=json.dumps({**diag, "rmse_m": fit.rmse_m,
                              "n_inliers": fit.n_inliers,
                              "z_range": fit.z_range}))
    log.info("Wrote %s (median depth %.2fm)", npz_path, diag["plate_depth_median_m"])

    # Base-frame cloud + a colourised depth for eyeballing.
    from o2m.splat.camera import Camera
    from o2m.splat.pointcloud import (colorize_depth, unproject_depth_to_points,
                                      write_ply)
    cam = Camera.from_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                 plate.shape[1], plate.shape[0], c2w)
    xyz, rgb = unproject_depth_to_points(cam, depth, plate, stride=2,
                                         max_depth=args.max_depth)
    ply = write_ply(xyz, rgb, out_root / "pointcloud" / "zed_scene_metric.ply")
    log.info("Wrote %s (%d points)", ply, len(xyz))

    vis = colorize_depth(np.nan_to_num(depth, nan=0.0))
    vis_path = out_root / "renders" / "worldmodel" / "zed_scene_depth.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(vis_path)
    log.info("Wrote %s", vis_path)


if __name__ == "__main__":
    main()
