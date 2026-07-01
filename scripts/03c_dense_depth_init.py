#!/usr/bin/env python3
"""Stage 03c (optional) — monocular-depth dense initialisation for the splat.

Runs Depth-Anything-V2 on the wrist frames, aligns each to the COLMAP sparse
points, unprojects to a dense seed point cloud, and rewires transforms.json so
splatfacto initialises its Gaussians from it. This fixes geometry on textureless
surfaces (blank walls) that photometric SfM/splatting leaves under-constrained.

Run after stage 03 (poses) and before stage 04 (train). Optional layer: if it
fails or is skipped, training falls back to the sparse COLMAP seed.
"""
from __future__ import annotations

import json

from _common import base_parser, load, log

from o2m.depth import MonoDepthEstimator, build_dense_seed_cloud


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--pixel-stride", type=int, default=3)
    p.add_argument("--max-points", type=int, default=600_000)
    args = p.parse_args()
    cfg, paths = load(args)

    if not paths.colmap_sparse.exists():
        raise SystemExit("No COLMAP model; run 03_run_colmap.py first.")

    log.info("Loading Depth-Anything-V2 (%s) ...", args.model)
    estimator = MonoDepthEstimator(model=args.model)

    out_ply = paths.nerfstudio / "dense_seed.ply"
    result = build_dense_seed_cloud(
        sparse_dir=paths.colmap_sparse, frames_dir=paths.frames,
        depth_estimator=estimator, out_ply=out_ply,
        pixel_stride=args.pixel_stride, max_points=args.max_points,
    )
    if result is None:
        log.warning("Dense seed not produced; leaving the sparse COLMAP seed in place.")
        return

    # Rewire transforms.json to initialise the splat from the dense cloud.
    with open(paths.transforms_json) as f:
        transforms = json.load(f)
    transforms["ply_file_path"] = "dense_seed.ply"
    with open(paths.transforms_json, "w") as f:
        json.dump(transforms, f, indent=2)
    log.info("Splat seed set to dense cloud -> %s", out_ply)


if __name__ == "__main__":
    main()
