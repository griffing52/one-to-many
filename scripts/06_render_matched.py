#!/usr/bin/env python3
"""Stage 06 — render the splat at the dataset framerate, one frame per recorded
frame.

COLMAP only registers a subset of (subsampled) frames, so playing those back
stutters. This renders the splat at EVERY original recorded frame: exact poses
at registered keyframes, slerp/lerp-interpolated poses in between (the wrist
moves smoothly). Output is a side-by-side [original | splat] video at the
recording fps, plus a splat-only video, so each rendered frame matches the
recorded viewpoint.
"""
from __future__ import annotations

import json

import numpy as np

from _common import base_parser, load, log

from o2m.splat import SplatModel, Camera, interpolate_c2w
from o2m.render import video


def _fingerprint(path, size=32):
    import cv2
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (size, size)).astype(np.float32).ravel()


def resolve_raw_by_hash(transforms, images_dir, raw_frames):
    """Map each registered frame to its identical raw frame by image fingerprint.

    Robust to frame_index.json corruption (stale stride / orphan frames): the
    extracted frames are exact copies of raw frames, so nearest-fingerprint
    matching recovers the true raw index.
    """
    from pathlib import Path
    raw_fps = np.stack([_fingerprint(p) for p in raw_frames])
    mapping = {}
    for fr in transforms["frames"]:
        name = Path(fr["file_path"]).name
        fp = _fingerprint(Path(images_dir) / name)
        mapping[fr["file_path"]] = int(np.argmin(np.linalg.norm(raw_fps - fp, axis=1)))
    return mapping


def _pick_complete_config(splat_dir):
    """Newest config.yml that has a finished (29999) checkpoint."""
    cfgs = sorted(splat_dir.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cfgs:
        if list((c.parent / "nerfstudio_models").glob("step-000029999.ckpt")):
            return c
    return cfgs[0] if cfgs else None


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--splat-config", default=None, help="Path to a specific config.yml.")
    p.add_argument("--fps", type=int, default=None, help="Override dataset fps.")
    p.add_argument("--full-range", action="store_true",
                   help="Render all recorded frames (freezes outside the registered span).")
    p.add_argument("--range", type=int, nargs=2, default=None, metavar=("START", "END"),
                   help="Render only raw frames [START, END] (inclusive). Overrides the "
                        "default registered span; poses still interpolate from all keyframes.")
    p.add_argument("--map", choices=["frame_index", "hash"], default="frame_index",
                   help="How to map registered frames to raw indices. 'hash' is "
                        "robust to a stale/corrupt frame_index.json.")
    args = p.parse_args()
    cfg, paths = load(args)

    fps = args.fps or int(cfg.get("render.fps", 30))
    import cv2

    # --- registered keyframe poses on the raw frame timeline -----------------
    with open(paths.transforms_json) as f:
        transforms = json.load(f)
    wrist_dir = paths.raw_camera(cfg.get("dataset.cameras.wrist", "realsense_color"))
    raw_frames = sorted(wrist_dir.glob("*.png"), key=lambda p: p.stem)

    if args.map == "hash":
        log.info("Resolving raw indices by image fingerprint (robust mapping) ...")
        raw_idx_of = resolve_raw_by_hash(transforms, paths.frames, raw_frames)
    else:
        with open(paths.frames / "frame_index.json") as f:
            selected = json.load(f)["selected_raw_indices"]
        raw_idx_of = {fr["file_path"]: selected[int(fr["file_path"].split("/")[-1].split(".")[0])]
                      for fr in transforms["frames"]}

    key_raw, key_c2w = [], []
    for fr in sorted(transforms["frames"], key=lambda f: f["file_path"]):
        cam = Camera.from_transforms_frame({**transforms, "frames": [fr]}, 0)
        key_raw.append(raw_idx_of[fr["file_path"]])
        key_c2w.append(cam.c2w)
    # de-duplicate raw indices (hash collisions / repeats) keeping first
    order = np.argsort(key_raw)
    key_raw = np.array(key_raw)[order]
    key_c2w = np.array(key_c2w)[order]
    _, uniq = np.unique(key_raw, return_index=True)
    key_raw, key_c2w = key_raw[uniq], key_c2w[uniq]
    log.info("Registered keyframes: %d spanning raw frames %d..%d",
             len(key_raw), key_raw.min(), key_raw.max())

    # --- intrinsics ----------------------------------------------------------
    intr = Camera.from_transforms_frame(transforms, 0)

    # By default render only the registered span (outside it we have no poses, so
    # interpolation would just freeze the view). --full-range forces 0..N-1.
    lo, hi = int(key_raw.min()), int(key_raw.max())
    if args.full_range:
        lo, hi = 0, len(raw_frames) - 1
    if args.range:
        lo, hi = int(args.range[0]), min(int(args.range[1]), len(raw_frames) - 1)
    raw_frames = raw_frames[lo:hi + 1]
    query = np.arange(lo, hi + 1)
    poses = interpolate_c2w(key_raw, key_c2w, query)
    log.info("Rendering raw frames %d..%d (%d frames) at %d fps", lo, hi, len(query), fps)

    # --- render + assemble ----------------------------------------------------
    cfg_yml = args.splat_config or _pick_complete_config(paths.splat)
    if cfg_yml is None:
        raise SystemExit("No trained splat config found.")
    log.info("Rendering %d frames from %s", len(raw_frames), cfg_yml)
    splat = SplatModel.from_config(cfg_yml)

    splat_frames, sbs_frames = [], []
    for i, (pose, rawp) in enumerate(zip(poses, raw_frames)):
        cam = Camera(intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height, pose)
        rgb, _, _ = splat.render(cam)
        splat_frames.append(rgb)
        orig = cv2.cvtColor(cv2.imread(str(rawp)), cv2.COLOR_BGR2RGB)
        if orig.shape[:2] != rgb.shape[:2]:
            orig = cv2.resize(orig, (rgb.shape[1], rgb.shape[0]))
        sbs_frames.append(np.concatenate([orig, rgb], axis=1))
        if i % 50 == 0:
            log.info("  rendered %d/%d", i, len(raw_frames))

    out = paths.render_dir("matched", "dataset_fps")
    video.save_mp4(splat_frames, out / "splat_matched.mp4", fps=fps)
    video.save_mp4(sbs_frames, out / "side_by_side.mp4", fps=fps)
    log.info("Wrote %s (%d frames @ %d fps): splat_matched.mp4, side_by_side.mp4",
             out, len(splat_frames), fps)


if __name__ == "__main__":
    main()
