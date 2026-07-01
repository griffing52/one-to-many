#!/usr/bin/env python3
"""Orchestrate the full pipeline (stages from configs/pipeline.yaml `stages`).

Each stage shells out to the corresponding numbered script so a stage can also be
run/debugged standalone. Stages already completed can be skipped with --from.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _common import base_parser, load, log

_HERE = Path(__file__).resolve().parent
STAGE_SCRIPTS = {
    "extract": "01_extract_frames.py",
    "mask": "02_generate_masks.py",
    "colmap": "03_run_colmap.py",
    "align": "04b_align_world.py",
    "train": "04_train_splat.py",
    "render": "05_render_trajectory.py",
}


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--from", dest="from_stage", default=None,
                   help="Start at this stage (skip earlier ones).")
    p.add_argument("--max-iters", type=int, default=None,
                   help="Forwarded to the train stage for smoke runs.")
    args = p.parse_args()
    cfg, paths = load(args)

    stages = cfg.get("stages", list(STAGE_SCRIPTS))
    if args.from_stage:
        stages = stages[stages.index(args.from_stage):]

    for stage in stages:
        script = STAGE_SCRIPTS[stage]
        cmd = [sys.executable, str(_HERE / script),
               "--config", args.config]
        if args.episode:
            cmd += ["--episode", args.episode]
        if stage == "train" and args.max_iters:
            cmd += ["--max-iters", str(args.max_iters)]
        log.info("=== stage: %s -> %s ===", stage, script)
        subprocess.run(cmd, check=True)

    log.info("Pipeline complete for %s", paths.episode_id)


if __name__ == "__main__":
    main()
