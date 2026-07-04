#!/usr/bin/env python3
"""Stage 11 — fuse wrist video-depth into a static base-frame scene cloud.

Video-Depth-Anything over the (pre-grasp) wrist frames + FK ``hand_cam`` poses
+ one global metric scale anchored to the ZED scene (stage 10) -> a fused,
voxel-downsampled cloud of the static scene, merged with the ZED cloud.

    PYTHONPATH=src python scripts/11_fuse_wrist_cloud.py

Outputs:
  - outputs/<ep>/pointcloud/wrist_fused_metric.ply
  - outputs/<ep>/pointcloud/scene_merged_metric.ply  (ZED + wrist)
  - outputs/<ep>/align/wrist_scale.npz               (a, b, diagnostics)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from o2m.config import Config  # noqa: E402
from o2m.data import Episode, load_ee_trajectory, load_joint_trajectory  # noqa: E402
from o2m.robot import PiperModel  # noqa: E402
from o2m.utils import get_logger  # noqa: E402

log = get_logger("o2m.scripts.fuse")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worldmodel.yaml")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--frames", type=int, nargs=2, default=None,
                    metavar=("START", "END"),
                    help="Frame range to fuse; default = [0, grasp_frame).")
    ap.add_argument("--stride", type=int, default=3, help="Frame stride to fuse.")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--max-fuse-depth", type=float, default=1.2,
                    help="Fuse wrist points only nearer than this (m).")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    wm = cfg.section("worldmodel")
    root = Path(cfg.source).resolve().parent.parent

    def _abs(p):
        q = Path(p)
        return q if q.is_absolute() else (root / q).resolve()

    episode = args.episode or wm["episode"]
    ep = Episode(_abs(wm["data_root"]) / episode)
    df = ep.actions_df()
    arm = cfg.get("dataset.arm", "slave")
    joints = load_joint_trajectory(df, arm=arm)

    # Default fuse range: the static pre-grasp segment.
    if args.frames is not None:
        lo, hi = args.frames
    else:
        from o2m.worldmodel.perturb import detect_grasp_frame
        traj = load_ee_trajectory(df, arm=arm, source=cfg.get("dataset.ee_source", "ee"))
        lo, hi = 0, detect_grasp_frame(traj.gripper)
    idx = list(range(lo, hi, args.stride))
    log.info("Fusing frames [%d, %d) stride %d -> %d frames", lo, hi, args.stride, len(idx))

    from PIL import Image
    paths = ep.wrist_frames()
    frames = [np.asarray(Image.open(paths[i]).convert("RGB")) for i in idx]

    model = PiperModel(cfg.require("robot.urdf"), cfg.require("robot.urdf_dir"),
                       base_frame=cfg.get("robot.base_frame", "base_link"),
                       ee_frame=cfg.get("robot.ee_frame"),
                       camera_frame=cfg.get("robot.camera_frame") or "hand_cam")
    cam_frame = model.camera_frame
    from o2m.worldmodel.fusion import (align_clouds_icp,
                                       fit_wrist_scale_triangulated,
                                       fuse_wrist_cloud, optical_c2w)
    c2ws = [optical_c2w(model.fk(joints[i], [cam_frame])[cam_frame]) for i in idx]

    # Consistent disparity over the fused range.
    from o2m.depth import VideoDepthEstimator
    disps = VideoDepthEstimator(encoder=args.encoder).estimate_sequence(frames)

    # Metric anchor: the stage-10 ZED scene cloud (base frame).
    from o2m.splat.camera import Camera
    from o2m.splat.pointcloud import unproject_depth_to_points, write_ply
    sc = np.load(str(_abs("outputs") / episode / "align" / "zed_scene_metric.npz"))
    plate = np.asarray(Image.open(_abs(wm["clean_plate"])).convert("RGB"))
    K, c2w_zed = sc["K"], sc["c2w"]
    zed_cam = Camera.from_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                     plate.shape[1], plate.shape[0], c2w_zed)
    zed_xyz, zed_rgb = unproject_depth_to_points(zed_cam, sc["depth"], plate,
                                                 stride=2, max_depth=3.5)

    from o2m.worldmodel.wrist_warp import GripperMask, WristIntrinsics
    intr = WristIntrinsics(**wm["wrist_intrinsics"])
    gmask = GripperMask(**wm["gripper_mask"]).mask(intr.height, intr.width)

    # Metric scale from FK-triangulated feature tracks (self-contained; the
    # ZED-projection pairing decorrelates on this close-range low-texture view).
    fit = fit_wrist_scale_triangulated(frames, disps, c2ws, intr, gmask=gmask)
    depths = [fit.apply(d) for d in disps]

    # Fuse the NEAR field only: far white-wall depth is unconstrained in the
    # wrist view; the ZED cloud supplies the walls in the merged output.
    xyz, rgb = fuse_wrist_cloud(frames, depths, c2ws, intr, gmask=gmask,
                                z_range=(0.12, args.max_fuse_depth),
                                voxel=args.voxel)

    # The ZED c2w carries the PnP/f-prior calibration error; the wrist cloud is
    # FK-metric. ICP the ZED cloud onto the wrist cloud (overlap: rack + table)
    # before merging, so novel views don't ghost.
    T_icp, icp_diag = align_clouds_icp(zed_xyz, xyz)
    zed_xyz = zed_xyz @ T_icp[:3, :3].T + T_icp[:3, 3]

    out_root = root / "outputs" / episode
    np.savez(out_root / "align" / "wrist_scale.npz", a=fit.a, b=fit.b,
             rmse_m=fit.rmse_m, n_inliers=fit.n_inliers, frames=np.array(idx),
             T_icp_zed=T_icp)
    ply = write_ply(xyz, rgb, out_root / "pointcloud" / "wrist_fused_metric.ply")
    log.info("Wrote %s", ply)

    merged = write_ply(np.concatenate([zed_xyz, xyz]),
                       np.concatenate([zed_rgb, rgb]),
                       out_root / "pointcloud" / "scene_merged_metric.ply")
    log.info("Wrote %s (%d ZED + %d wrist points)", merged, len(zed_xyz), len(xyz))


if __name__ == "__main__":
    main()
