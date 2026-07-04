#!/usr/bin/env python3
"""Stage 13 — third-person renders from NEW camera positions/angles.

Renders the metric base-frame scene cloud (stage 10) from novel viewpoints and
depth-composites the URDF robot into each view. This answers "can we move the
third-person camera?": the ZED never moved, but the metric cloud + calibrated
robot let us re-shoot the scene from anywhere (quality degrades with distance
from the real viewpoint — unseen surfaces become holes).

    PYTHONPATH=src MUJOCO_GL=egl python scripts/13_novel_thirdperson.py

Outputs (outputs/<ep>/renders/novel_view/):
  - orbit_grasp.mp4        camera orbit at the grasp-frame joint config
  - episode_novel_view.mp4 whole episode replayed from one shifted viewpoint
  - angles_grid.png        sample stills at several azimuth/elevation offsets
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
from o2m.utils import get_logger  # noqa: E402

log = get_logger("o2m.scripts.novel")


def orbit_pose(base_c2w: np.ndarray, target: np.ndarray, azim_deg: float,
               elev_deg: float, radius_scale: float = 1.0) -> np.ndarray:
    """Rotate the camera position about the vertical axis through ``target``
    (base frame, z-up) and re-aim at ``target``."""
    from o2m.splat.camera import Camera
    eye0 = base_c2w[:3, 3]
    v = eye0 - target
    az = np.radians(azim_deg)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0], [0, 0, 1.0]])
    v = Rz @ v
    v = v * radius_scale
    v[2] += np.linalg.norm(v[:2]) * np.tan(np.radians(elev_deg))
    cam = Camera(1, 1, 0, 0, 1, 1, np.eye(4)).look_at(target + v, target)
    return cam.c2w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worldmodel.yaml")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--cloud", default=None,
                    help="Point cloud .ply (default: zed_scene_metric.ply).")
    ap.add_argument("--orbit-deg", type=float, default=30.0,
                    help="Max azimuth offset of the orbit (degrees).")
    ap.add_argument("--n-orbit", type=int, default=72)
    ap.add_argument("--novel-azim", type=float, default=18.0,
                    help="Azimuth offset for the episode replay view (deg).")
    ap.add_argument("--novel-elev", type=float, default=8.0)
    ap.add_argument("--episode-stride", type=int, default=2)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    wm = cfg.section("worldmodel")
    root = Path(cfg.source).resolve().parent.parent

    def _abs(p):
        q = Path(p)
        return q if q.is_absolute() else (root / q).resolve()

    episode = args.episode or wm["episode"]
    out_root = root / "outputs" / episode
    out_dir = out_root / "renders" / "novel_view"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scene cloud + real camera.
    cloud_path = Path(args.cloud) if args.cloud else \
        out_root / "pointcloud" / "zed_scene_metric.ply"
    data = np.loadtxt(cloud_path, skiprows=10)
    xyz, rgb = data[:, :3], data[:, 3:6].astype(np.uint8)
    log.info("Cloud: %s (%d points)", cloud_path.name, len(xyz))

    ext = np.load(str(_abs(wm["zed_extrinsic_npz"])))
    K, c2w0 = ext["K"], ext["c2w"]
    from o2m.splat.camera import Camera
    def cam_at(c2w):
        return Camera.from_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                      1280, 720, c2w)

    # Look-at target: centroid of the near scene (the table/rack working area).
    near = xyz[np.linalg.norm(xyz - c2w0[:3, 3], axis=1) < 1.6]
    target = near.mean(0) if len(near) else xyz.mean(0)
    log.info("Orbit target (base frame): %s", np.round(target, 3))

    ep = Episode(_abs(wm["data_root"]) / episode)
    df = ep.actions_df()
    arm = cfg.get("dataset.arm", "slave")
    joints = load_joint_trajectory(df, arm=arm)
    from o2m.worldmodel.perturb import detect_grasp_frame
    traj = load_ee_trajectory(df, arm=arm, source=cfg.get("dataset.ee_source", "ee"))
    grasp = detect_grasp_frame(traj.gripper)

    from o2m.render.points import render_points
    from o2m.render.composite import composite_rgba_over
    from o2m.robot import RobotRenderer
    rr = RobotRenderer(str(cfg.get("robot.render_urdf") or cfg.require("robot.urdf")),
                       cfg.require("robot.urdf_dir"))
    margin = float(wm.get("thirdperson", {}).get("depth_margin", 0.12))

    def shoot(c2w, q):
        cam = cam_at(c2w)
        bg, bg_depth, _ = render_points(xyz, rgb, cam, kernel=2, fill_method="inpaint")
        fg, alpha, fg_depth = rr.render_rgba(q, cam)
        return composite_rgba_over(bg, fg, alpha,
                                   fg_depth=fg_depth - margin, bg_depth=bg_depth)

    from o2m.render.video import save_mp4

    # 1. Orbit at the grasp config: azimuth sweep -orbit..+orbit and back.
    log.info("Orbit render (%d views, +-%.0f deg) ...", args.n_orbit, args.orbit_deg)
    sweep = np.sin(np.linspace(0, 2 * np.pi, args.n_orbit, endpoint=False))
    frames = [shoot(orbit_pose(c2w0, target, a * args.orbit_deg,
                               abs(a) * args.novel_elev), joints[grasp])
              for a in sweep]
    save_mp4(frames, out_dir / "orbit_grasp.mp4", fps=24)
    log.info("Wrote %s", out_dir / "orbit_grasp.mp4")

    # 2. Whole episode from ONE novel viewpoint (fixed camera => model cached).
    log.info("Episode replay from (azim %+.0f, elev %+.0f) ...",
             args.novel_azim, args.novel_elev)
    c2w_novel = orbit_pose(c2w0, target, args.novel_azim, args.novel_elev)
    cam = cam_at(c2w_novel)
    bg, bg_depth, _ = render_points(xyz, rgb, cam, kernel=2, fill_method="inpaint")
    ep_frames = []
    for i in range(0, len(joints), args.episode_stride):
        fg, alpha, fg_depth = rr.render_rgba(joints[i], cam)
        ep_frames.append(composite_rgba_over(bg, fg, alpha,
                                             fg_depth=fg_depth - margin,
                                             bg_depth=bg_depth))
    save_mp4(ep_frames, out_dir / "episode_novel_view.mp4", fps=30 // args.episode_stride)
    log.info("Wrote %s", out_dir / "episode_novel_view.mp4")

    # 3. Still grid over azimuth x elevation.
    from PIL import Image
    stills = []
    for elev in (0.0, 12.0):
        row = [shoot(orbit_pose(c2w0, target, az, elev), joints[grasp])
               for az in (-25.0, 0.0, 25.0)]
        stills.append(np.concatenate(row, axis=1))
    grid = np.concatenate(stills, axis=0)
    Image.fromarray(grid).save(out_dir / "angles_grid.png")
    log.info("Wrote %s", out_dir / "angles_grid.png")


if __name__ == "__main__":
    main()
