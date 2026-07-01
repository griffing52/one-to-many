#!/usr/bin/env python3
"""Stage 04b — solve the splat-world <-> robot-base similarity (sim3).

For each reconstructed wrist frame we pair its COLMAP camera centre (splat frame)
with the FK camera-link centre at the measured joints (base frame), then Umeyama-
fit the sim3. Writes outputs/<ep>/align/sim3.json. See docs/alignment.md.
"""
from __future__ import annotations

import json

import numpy as np

from _common import base_parser, load, log

from o2m.align import WorldAligner
from o2m.data import Episode, load_joint_trajectory
from o2m.splat.camera import Camera
from o2m.robot import PiperModel


def main() -> None:
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg, paths = load(args)

    with open(paths.transforms_json) as f:
        transforms = json.load(f)
    with open(paths.frames / "frame_index.json") as f:
        selected = json.load(f)["selected_raw_indices"]

    # COLMAP camera centres in the splat frame, ordered by file_path.
    frames = sorted(transforms["frames"], key=lambda fr: fr["file_path"])
    colmap_centers, kept_pos = [], []
    for fr in frames:
        # frame file_path is images/000123.png -> position in the extracted set
        pos = int(fr["file_path"].split("/")[-1].split(".")[0])
        cam = Camera.from_transforms_frame({**transforms, "frames": [fr]}, 0)
        colmap_centers.append(cam.c2w[:3, 3])
        kept_pos.append(pos)
    colmap_centers = np.array(colmap_centers)

    # FK camera centres in the base frame at the matching measured joints.
    ep = Episode(paths.raw_episode,
                 wrist_dir=cfg.get("dataset.cameras.wrist", "realsense_color"),
                 zed_dir=cfg.get("dataset.cameras.zed", "zed_color"))
    joints = load_joint_trajectory(ep.actions_df(), arm=cfg.get("dataset.arm", "slave"))

    model = PiperModel(cfg.require("robot.urdf"), cfg.require("robot.urdf_dir"),
                       base_frame=cfg.get("robot.base_frame", "base_link"),
                       ee_frame=cfg.get("robot.ee_frame"),
                       camera_frame=cfg.get("robot.camera_frame"))

    fk_centers = []
    for pos in kept_pos:
        raw_i = selected[pos]
        fk_centers.append(model.camera_pose_base(joints[raw_i])[:3, 3])
    fk_centers = np.array(fk_centers)

    sim3, diag = WorldAligner.from_wrist_fk(colmap_centers, fk_centers)
    sim3.to_json(paths.sim3_json)
    log.info("sim3 solved: scale=%.4f  residual_rms=%.4f m  (n=%d) -> %s",
             diag["scale"], diag["residual_rms"], diag["n"], paths.sim3_json)
    if diag["residual_rms"] > 0.05:
        log.warning("High alignment residual (%.3f m). Check masks/FK camera link "
                    "offset; see docs/alignment.md.", diag["residual_rms"])


if __name__ == "__main__":
    main()
