#!/usr/bin/env python3
"""Stage 12 — MVGenMaster multi-view synthesis: conditioning-strategy shootout.

Synthesizes ONE perturbed wrist view (frame ``t`` shifted by ``--offset`` in the
base frame) with MVGenMaster under different sets of reference views, all with
KNOWN cameras (wrist = FK of ``hand_cam``, third-person = calibrated ZED):

  self      : [t]                      single-ref baseline
  pm_k      : [t-k, t, t+k]            temporal neighbours
  pm_2k     : [t-2k, t-k, t, t+k, t+2k]
  nbrs_only : [t-k, t+k]               can neighbours alone recreate t+offset?
  zed       : [t, ZED clean plate]     third-person as the second view
  zed_k     : [t-k, t, t+k, ZED]       mixture

Reference depth comes from DUSt3R locked to the metric FK poses (default,
geometrically consistent) or Video-Depth-Anything (``--depth vda``, the
depth-warp's pseudo-metric depth + ``--depth-scale`` fudge). The ZED ref always
uses the metric plate depth (``zed_scene_metric.npz``).

Output: one labelled montage + per-strategy PNGs under renders/worldmodel/,
with the real frame and the depth-warp of the SAME offset as baselines.

Example
-------
    PYTHONPATH=src python scripts/12_mvgen_demo.py \
        --frame 40 --offset 0 0.04 0.02 --k 5 --strategies self,pm_k,zed
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
from o2m.worldmodel import GripperMask, WristIntrinsics, WristWarper  # noqa: E402
from o2m.worldmodel.mvgen_warp import (MVGenWrapper, crop_to_aspect,  # noqa: E402
                                       offset_target_c2w, wrist_c2w)

STRATEGIES = ("self", "pm_k", "pm_2k", "nbrs_only", "zed", "zed_k")


def _abs(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q).resolve()


def _label(img, text):
    im = Image.fromarray(np.asarray(img, np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 12, 20], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


def _ref_frames(strategy: str, t: int, k: int, n: int):
    """Wrist frame indices for a strategy (+ whether the ZED plate is a ref)."""
    idx = {
        "self": [t],
        "pm_k": [t - k, t, t + k],
        "pm_2k": [t - 2 * k, t - k, t, t + k, t + 2 * k],
        "nbrs_only": [t - k, t + k],
        "zed": [t],
        "zed_k": [t - k, t, t + k],
    }[strategy]
    idx = [i for i in idx if 0 <= i < n]
    return idx, strategy in ("zed", "zed_k")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worldmodel.yaml")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.04, 0.02])
    ap.add_argument("--k", type=int, default=5, help="neighbour stride (frames)")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--depth", default="dust3r", choices=["dust3r", "vda"])
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="ref-depth multiplier (vda depth is pseudo-metric)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    wm = cfg.section("worldmodel")
    root = Path(cfg.source).resolve().parent.parent
    ep = Episode(_abs(root, wm["data_root"]) / wm["episode"])
    wi = WristIntrinsics(**wm["wrist_intrinsics"])
    gm = GripperMask(**wm["gripper_mask"])
    K_wrist = np.array([[wi.fx, 0, wi.cx], [0, wi.fy, wi.cy], [0, 0, 1.0]])

    model = PiperModel(cfg.require("robot.urdf"), cfg.require("robot.urdf_dir"),
                       camera_frame=cfg.get("robot.camera_frame") or "hand_cam")
    mj = load_joint_trajectory(ep.actions_df())
    wrist_paths = ep.wrist_frames()
    n = min(len(mj), len(wrist_paths))
    t = args.frame
    offset = np.array(args.offset)

    c2w = {i: wrist_c2w(model.fk(mj[i], ["hand_cam"])["hand_cam"])
           for i in range(max(0, t - 2 * args.k), min(n, t + 2 * args.k + 1))}
    tar_c2w = offset_target_c2w(c2w[t], offset)
    real_t = np.asarray(Image.open(wrist_paths[t]).convert("RGB"))

    # ZED third-person ref: clean plate + calibrated camera + METRIC plate depth,
    # centre-cropped to the wrist aspect ratio (one resolution per model call).
    zed = np.load(str(_abs(root, wm["zed_extrinsic_npz"])))
    plate = np.asarray(Image.open(_abs(root, wm["clean_plate"])).convert("RGB"))
    from o2m.worldmodel.scene_cloud import load_scene_depth
    zed_depth = load_scene_depth(_abs(root, wm["thirdperson"]["scene_depth_npz"]))
    if zed_depth is None:
        raise SystemExit("zed_scene_metric.npz missing - run scripts/10_zed_metric_scene.py")
    zed_rgb, zed_K, zed_dep = crop_to_aspect(plate, zed["K"],
                                             wi.height / wi.width, depth=zed_depth)

    mv = MVGenWrapper(num_inference_steps=args.steps, guidance_scale=args.guidance,
                      seed=args.seed)

    # depth for every wrist frame any strategy needs
    need = sorted({t} | {i for s in args.strategies.split(",")
                         for i in _ref_frames(s, t, args.k, n)[0]})
    rgbs = {i: np.asarray(Image.open(wrist_paths[i]).convert("RGB")) for i in need}
    print(f"wrist ref depth ({args.depth}) for frames {need} ...")
    if args.depth == "dust3r":
        if len(need) > 1:
            deps = mv.dust3r_depths([rgbs[i] for i in need], [c2w[i] for i in need],
                                    [K_wrist] * len(need))
        else:  # dust3r needs pairs; single frame falls back to VDA depth
            deps = _vda_depths([rgbs[i] for i in need])
        depth = dict(zip(need, deps))
    else:
        depth = dict(zip(need, _vda_depths([rgbs[i] for i in need])))

    gmask = gm.mask(wi.height, wi.width)
    panels = [_label(real_t, f"real f{t}"),
              _label(np.asarray(Image.open(wrist_paths[t + args.k]).convert("RGB")),
                     f"real f{t + args.k} (ref)")]

    # depth-warp baseline with the SAME depth + offset. c2w rotation is
    # base<-optical, so its transpose maps the base offset into the optical frame
    # (identical to base_offset_to_camera on the FK link rotation).
    dcam = c2w[t][:3, :3].T @ offset
    warper = WristWarper(wi, gm, kernel_splat=True)
    dw = warper.warp(real_t, depth[t] * args.depth_scale, dcam)
    panels.append(_label(dw, f"depthwarp {args.depth}"))

    results = {}
    for s in args.strategies.split(","):
        idx, use_zed = _ref_frames(s, t, args.k, n)
        ref_rgbs = [rgbs[i] for i in idx]
        ref_deps = [depth[i] * args.depth_scale for i in idx]
        ref_c2ws = [c2w[i] for i in idx]
        ref_Ks = [K_wrist] * len(idx)
        if use_zed:
            ref_rgbs.append(zed_rgb)
            ref_deps.append(zed_dep)
            ref_c2ws.append(zed["c2w"])
            ref_Ks.append(zed_K)
        t0 = time.time()
        out = mv.synthesize(ref_rgbs, ref_deps, ref_c2ws, ref_Ks, [tar_c2w],
                            out_size=(wi.height, wi.width))[0]
        dt = time.time() - t0
        out[gmask] = real_t[gmask]        # gripper is rigid to the camera
        results[s] = out
        print(f"  {s:10s} refs={len(ref_rgbs)}  {dt:.1f}s")
        panels.append(_label(out, f"mvgen {s} {dt:.0f}s"))

    cols = 4
    while len(panels) % cols:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.concatenate(panels[r:r + cols], axis=1)
            for r in range(0, len(panels), cols)]
    grid = np.concatenate(rows, axis=0)
    out_dir = Path(args.out).parent if args.out else (
        _abs(root, wm["output_root"]).parent / "renders" / "worldmodel")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (
        out_dir / f"mvgen_strategies_f{t}_{args.depth}.png")
    Image.fromarray(grid).save(out_path)
    for s, im in results.items():
        Image.fromarray(im).save(out_dir / f"mvgen_f{t}_{s}_{args.depth}.png")
    print("saved", out_path)


def _vda_depths(frames):
    """Video-Depth-Anything depth (pseudo-metric, median 0.5m) for a frame list."""
    from o2m.depth import VideoDepthEstimator
    from o2m.worldmodel.wrist_warp import disparities_to_depths
    vde = VideoDepthEstimator()
    return list(disparities_to_depths(vde.estimate_sequence(list(frames))))


if __name__ == "__main__":
    main()
