#!/usr/bin/env python3
"""Stage 01 — extract wrist frames for reconstruction.

Reads the raw episode, copies (subsampled) wrist PNGs into
outputs/<ep>/frames/ as COLMAP's image set, and records which raw frame indices
were selected (needed later to align measured joints with the COLMAP poses).
"""
from __future__ import annotations

import json
import shutil

from _common import base_parser, load, log

from o2m.data import Episode


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--frame-stride", type=int, default=None)
    p.add_argument("--start-frame", type=int, default=0,
                   help="First raw frame to include (inclusive).")
    p.add_argument("--end-frame", type=int, default=None,
                   help="Last raw frame to include (exclusive); default = end.")
    args = p.parse_args()
    cfg, paths = load(args)

    stride = args.frame_stride or int(cfg.get("dataset.frame_stride", 1))
    wrist_dir = cfg.get("dataset.cameras.wrist", "realsense_color")
    zed_dir = cfg.get("dataset.cameras.zed", "zed_color")

    ep = Episode(paths.raw_episode, wrist_dir=wrist_dir, zed_dir=zed_dir)
    ep.validate()
    frames = ep.wrist_frames()
    end = args.end_frame if args.end_frame is not None else len(frames)
    selected = list(range(args.start_frame, min(end, len(frames)), stride))

    # Clear any previous extraction so a different stride can't leave orphan
    # frames behind (which would corrupt the COLMAP set and frame_index mapping).
    for old in paths.frames.glob("*.png"):
        old.unlink()

    for new_i, raw_i in enumerate(selected):
        dst = paths.frames / f"{new_i:06d}.png"
        shutil.copy2(frames[raw_i], dst)

    index = {"stride": stride, "selected_raw_indices": selected,
             "num_extracted": len(selected), "wrist_dir": wrist_dir}
    with open(paths.frames / "frame_index.json", "w") as f:
        json.dump(index, f, indent=2)

    log.info("Extracted %d/%d wrist frames -> %s", len(selected), len(frames), paths.frames)


if __name__ == "__main__":
    main()
