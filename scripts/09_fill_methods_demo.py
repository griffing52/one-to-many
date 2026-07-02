#!/usr/bin/env python3
"""Stage 09 — compare wrist-view hole-fill / novel-view methods on one frame.

Warps one recorded wrist frame by a chosen offset and shows every disocclusion
fill strategy side-by-side, plus (if the checkpoints are present) GenWarp's
generative novel view. Use it to pick a `fill_method`/`wrist_renderer` for
`configs/worldmodel.yaml`.

Panels: original | raw scatter (no fill) | nearest | bilinear | edge_aware |
        inpaint(TELEA) | GenWarp warped | GenWarp synthesized

Example
-------
    PYTHONPATH=src MUJOCO_GL=egl python scripts/09_fill_methods_demo.py \
        --frame 40 --offset 0 0.08 0.04 --out outputs/.../fill_methods.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from o2m.config import Config  # noqa: E402
from o2m.data import Episode, load_joint_trajectory  # noqa: E402
from o2m.robot import PiperModel  # noqa: E402
from o2m.worldmodel import (GripperMask, WristIntrinsics, WristWarper,  # noqa: E402
                            base_offset_to_camera, disparity_to_depth)
from o2m.depth import MonoDepthEstimator  # noqa: E402


def _abs(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q).resolve()


def _label(img, text):
    im = Image.fromarray(np.asarray(img, np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 12, 20], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worldmodel.yaml")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.08, 0.04])
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-genwarp", action="store_true")
    ap.add_argument("--genwarp-mode", default="pad", choices=["pad", "crop", "squash"])
    ap.add_argument("--depth-scale", type=float, default=1.0)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    wm = cfg.section("worldmodel")
    root = Path(cfg.source).resolve().parent.parent
    ep = Episode(_abs(root, wm["data_root"]) / wm["episode"])
    wi = WristIntrinsics(**wm["wrist_intrinsics"])
    gm = GripperMask(**wm["gripper_mask"])

    # dcam at this frame from FK(measured) + base offset (full, no envelope).
    model = PiperModel(cfg.require("robot.urdf"), cfg.require("robot.urdf_dir"),
                       camera_frame=cfg.get("robot.camera_frame") or "hand_cam")
    mj = load_joint_trajectory(ep.actions_df())
    cam_R = model.fk(mj[args.frame], ["hand_cam"])["hand_cam"][:3, :3]
    dcam = base_offset_to_camera(np.array(args.offset), cam_R)

    real = np.asarray(Image.open(ep.wrist_frames()[args.frame]).convert("RGB"))
    print("estimating depth ...")
    depth = disparity_to_depth(MonoDepthEstimator().estimate(real))

    warper = WristWarper(wi, gm, kernel_splat=True)
    scat, filled = warper.scatter(real, depth, dcam)
    hole_pct = 100.0 * (~filled).mean()
    print(f"disocclusion holes: {hole_pct:.1f}% of pixels")

    panels = [_label(real, "original")]
    from o2m.worldmodel.wrist_warp import fill_holes
    for method in ("none", "nearest", "bilinear", "edge_aware", "inpaint"):
        t0 = time.time()
        out = fill_holes(scat.copy(), filled, method)
        dt = time.time() - t0
        out = out.copy(); out[gm.mask(wi.height, wi.width)] = real[gm.mask(wi.height, wi.width)]
        panels.append(_label(out, f"{method} {dt*1000:.0f}ms"))

    if not args.no_genwarp:
        try:
            from o2m.worldmodel.genwarp_warp import GenWarpWrapper
            gw = GenWarpWrapper()
            t0 = time.time()
            aux = gw.warp(real, depth, dcam, wi.fy, return_aux=True,
                          mode=args.genwarp_mode, depth_scale=args.depth_scale)
            dt = time.time() - t0
            syn = aux["synthesized"].copy()
            syn[gm.mask(wi.height, wi.width)] = real[gm.mask(wi.height, wi.width)]
            panels.append(_label(aux["warped"], "genwarp warped"))
            panels.append(_label(syn, f"genwarp synth {dt:.1f}s"))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[genwarp skipped] {type(e).__name__}: {str(e)[:120]}")

    # grid: 4 columns
    cols = 4
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.concatenate(panels[r:r + cols], axis=1) for r in range(0, len(panels), cols)]
    grid = np.concatenate(rows, axis=0)
    out_path = Path(args.out) if args.out else (
        _abs(root, wm["output_root"]).parent / "renders" / "worldmodel"
        / f"fill_methods_f{args.frame}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print("saved", out_path)


if __name__ == "__main__":
    main()
