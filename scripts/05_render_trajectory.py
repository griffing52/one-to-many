#!/usr/bin/env python3
"""Stage 05 — render a (possibly altered) trajectory from the splat.

Modes:
  static         orbit a free camera through the environment splat.
  robot_overlay  render the splat from a fixed viewpoint and composite the URDF
                 arm at the edited trajectory (measured joints for replay,
                 IK otherwise), mapped base->splat via the sim3.
  dynamic        robot_overlay + re-posed object (stretch; see docs).
"""
from __future__ import annotations

import json

import numpy as np

from _common import base_parser, load, log

from o2m.align import Sim3
from o2m.data import Episode, load_ee_trajectory, load_joint_trajectory
from o2m.render import RenderPipeline, video
from o2m.splat import SplatModel
from o2m.splat.camera import Camera, orbit_cameras
from o2m.trajectory import apply_preset


def _find_config(splat_dir):
    cands = sorted(splat_dir.rglob("config.yml"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("No trained splat config.yml; run 04_train_splat.py first.")
    return cands[-1]


def _viewpoint_camera(cfg, paths, transforms) -> Camera:
    """Pick the fixed render camera (in the splat frame)."""
    which = cfg.get("render.viewpoint", "wrist")
    zed_json = paths.align / "zed_camera.json"
    if which == "zed" and zed_json.exists():
        d = json.loads(zed_json.read_text())
        return Camera.from_intrinsics(d["fx"], d["fy"], d["cx"], d["cy"],
                                      d["width"], d["height"], np.array(d["c2w"]))
    if which == "zed":
        log.warning("No registered ZED camera (%s); falling back to a wrist view. "
                    "Register the ZED into the splat frame to use the demo viewpoint.",
                    zed_json)
    return Camera.from_transforms_frame(transforms, frame_index=0)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--mode", default=None, choices=["static", "robot_overlay", "dynamic"])
    p.add_argument("--edit", default=None, help="Trajectory preset name.")
    args = p.parse_args()
    cfg, paths = load(args)

    mode = args.mode or cfg.get("render.mode", "robot_overlay")
    edit = args.edit or cfg.get("render.edit", "replay_original")
    fps = int(cfg.get("render.fps", 30))

    with open(paths.transforms_json) as f:
        transforms = json.load(f)

    splat = SplatModel.from_config(_find_config(paths.splat))
    out_dir = paths.render_dir(mode, edit)

    if mode == "static":
        base = Camera.from_transforms_frame(transforms, 0)
        center = np.mean([Camera.from_transforms_frame({**transforms, "frames": [fr]}, 0).c2w[:3, 3]
                          for fr in transforms["frames"]], axis=0)
        cams = orbit_cameras(base, center, radius=0.6, n=fps * 4)
        frames = RenderPipeline(splat).render_static(cams)
    else:
        from o2m.robot import PiperModel, RobotRenderer

        sim3 = Sim3.from_json(paths.sim3_json)
        model = PiperModel(cfg.require("robot.urdf"), cfg.require("robot.urdf_dir"),
                           base_frame=cfg.get("robot.base_frame", "base_link"),
                           ee_frame=cfg.get("robot.ee_frame"),
                           camera_frame=cfg.get("robot.camera_frame"))
        renderer = RobotRenderer(
            cfg.get("robot.render_urdf", cfg.require("robot.urdf")),
            cfg.require("robot.urdf_dir"))
        pipe = RenderPipeline(splat, robot_renderer=renderer, robot_model=model, sim3=sim3)

        ep = Episode(paths.raw_episode,
                     wrist_dir=cfg.get("dataset.cameras.wrist", "realsense_color"),
                     zed_dir=cfg.get("dataset.cameras.zed", "zed_color"))
        df = ep.actions_df()
        traj = load_ee_trajectory(df, arm=cfg.get("dataset.arm", "slave"),
                                  source=cfg.get("dataset.ee_source", "ee"))
        edited = apply_preset(edit, traj)

        if edit == "replay_original":
            joints = load_joint_trajectory(df, arm=cfg.get("dataset.arm", "slave"))
        else:
            joints = pipe.ee_traj_to_joints(edited)

        # Express the splat-frame viewpoint in the metric base frame for the arm.
        vp_splat = _viewpoint_camera(cfg, paths, transforms)
        vp_base = Camera(vp_splat.fx, vp_splat.fy, vp_splat.cx, vp_splat.cy,
                         vp_splat.width, vp_splat.height, sim3.inv_apply(vp_splat.c2w))

        bg, bg_depth, _ = splat.render(vp_splat)
        frames = []
        from o2m.render.composite import composite_rgba_over
        for q in joints:
            fg, alpha, _ = renderer.render_rgba(q, vp_base)
            frames.append(composite_rgba_over(bg, fg, alpha))

    video.save_frames_png(frames, out_dir)
    video.save_mp4(frames, out_dir / "render.mp4", fps=fps)
    log.info("Rendered %d frames (mode=%s edit=%s) -> %s", len(frames), mode, edit, out_dir)


if __name__ == "__main__":
    main()
