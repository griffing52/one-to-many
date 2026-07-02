#!/usr/bin/env python3
"""Stage 07 — synthesize a perturbed episode (cheap world model).

Reads ``configs/worldmodel.yaml``, perturbs one recorded episode with a converging
offset (reaches the same grasp, ends the same way), checks IK feasibility, renders
the WRIST (depth-warp) and THIRD-PERSON (URDF robot over the ZED clean plate)
views, and writes a synthetic episode back in the input dataset format.

Examples
--------
    # Full episode, both views, using the offset in the config:
    PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py

    # Quick smoke test: 20 frames around the grasp, override the offset & name:
    PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py \
        --frames 120 141 --offset 0 0.06 0.03 --name left_up_6_3cm

Outputs -> worldmodel.output_root/<perturb name>/  (actions.csv, realsense_color/,
zed_color/, perturbation.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from o2m.config import Config  # noqa: E402
from o2m.utils import get_logger  # noqa: E402
from o2m.worldmodel import (GripperMask, PerturbationSpec, SyntheticEpisodePipeline,  # noqa: E402
                            WorldModelConfig, WristIntrinsics)

log = get_logger("o2m.scripts.synth")


def _abs(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (root / q).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/worldmodel.yaml")
    ap.add_argument("--episode", default=None, help="Override worldmodel.episode.")
    ap.add_argument("--offset", type=float, nargs=3, default=None,
                    metavar=("DX", "DY", "DZ"), help="Override base_offset (m).")
    ap.add_argument("--envelope", default=None,
                    choices=["bump", "converge_at_grasp", "constant"])
    ap.add_argument("--grasp-frame", type=int, default=None)
    ap.add_argument("--name", default=None, help="Perturbation name -> output subdir.")
    ap.add_argument("--frames", type=int, nargs=2, default=None,
                    metavar=("START", "END"), help="Frame range [start, end).")
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--no-zed", action="store_true")
    ap.add_argument("--wrist-renderer", default=None, choices=["depthwarp", "genwarp"])
    ap.add_argument("--fill-method", default=None,
                    choices=["none", "nearest", "bilinear", "edge_aware", "inpaint"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    wm = cfg.section("worldmodel")
    root = Path(cfg.source).resolve().parent.parent  # project root (configs/..)

    episode = args.episode or wm["episode"]
    episode_dir = _abs(root, wm["data_root"]) / episode

    p = wm["perturb"]
    spec = PerturbationSpec(
        base_offset=tuple(args.offset if args.offset is not None else p["base_offset"]),
        envelope=args.envelope or p.get("envelope", "bump"),
        grasp_frame=args.grasp_frame if args.grasp_frame is not None else p.get("grasp_frame"),
        name=args.name or p.get("name", "perturb"))

    wi = wm["wrist_intrinsics"]
    gm = wm["gripper_mask"]
    warp = wm.get("warp", {})
    gwc = wm.get("genwarp", {})
    fr = args.frames if args.frames is not None else wm.get("frame_range")

    wmc = WorldModelConfig(
        episode_dir=episode_dir,
        urdf=Path(cfg.require("robot.urdf")),
        render_urdf=Path(cfg.get("robot.render_urdf", cfg.require("robot.urdf"))),
        urdf_dir=Path(cfg.require("robot.urdf_dir")),
        zed_extrinsic_npz=_abs(root, wm["zed_extrinsic_npz"]),
        clean_plate=_abs(root, wm["clean_plate"]),
        wrist_intr=WristIntrinsics(**wi),
        gripper_mask=GripperMask(**gm),
        spec=spec,
        output_dir=_abs(root, wm["output_root"]) / spec.name,
        ee_frame=cfg.get("robot.ee_frame"),
        camera_frame=cfg.get("robot.camera_frame") or "hand_cam",
        base_frame=cfg.get("robot.base_frame", "base_link"),
        arm=cfg.get("dataset.arm", "slave"),
        ee_source=cfg.get("dataset.ee_source", "ee"),
        ik_tol=float(p.get("ik_tol", 5e-3)),
        kernel_splat=bool(warp.get("kernel_splat", True)),
        inpaint_holes=bool(warp.get("inpaint_holes", True)),
        fill_method=args.fill_method or warp.get("fill_method", "inpaint"),
        wrist_renderer=args.wrist_renderer or wm.get("wrist_renderer", "depthwarp"),
        genwarp_mode=gwc.get("mode", "pad"),
        genwarp_depth_scale=float(gwc.get("depth_scale", 1.0)),
        genwarp_steps=int(gwc.get("num_inference_steps", 20)),
        genwarp_guidance=float(gwc.get("guidance_scale", 3.5)),
        frame_range=tuple(fr) if fr else None,
        render_wrist=wm.get("render_wrist", True) and not args.no_wrist,
        render_zed=wm.get("render_zed", True) and not args.no_zed,
    )

    out = SyntheticEpisodePipeline(wmc).run()
    log.info("Done -> %s", out)


if __name__ == "__main__":
    main()
