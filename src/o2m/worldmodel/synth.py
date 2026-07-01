"""Write a synthetic episode in the SAME on-disk format as the input dataset.

Layout produced (identical to ``agilex_data_collection/pick_bag_joe/episode_NNN``)::

    <out_dir>/
      actions.csv            # same columns; slave_j*/slave_ee_* overwritten
      realsense_color/000000.png ...   (640x480 warped wrist frames)
      zed_color/000000.png ...         (1280x720 composited third-person frames)
      perturbation.json      # provenance: spec, grasp frame, feasibility, label

So downstream training/LeRobot conversion treats it exactly like a real episode.
Units match ``o2m.data.actions``: positions m -> 0.001mm (x1e6), joints rad ->
0.001deg (/ (pi/180/1000)).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..data.actions import MDEG_TO_RAD, MM_001_TO_M
from .perturb import FeasibilityReport, PerturbedTrajectory


def _write_pngs(dir_: Path, frames) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        Image.fromarray(np.asarray(fr, np.uint8)).save(dir_ / f"{i:06d}.png")


def write_synthetic_episode(out_dir: str | Path, base_df: pd.DataFrame,
                            pert: PerturbedTrajectory, feas: FeasibilityReport,
                            wrist_frames, zed_frames, arm: str = "slave",
                            source_episode: str = "") -> Path:
    """Write actions.csv + camera streams + metadata for one synthetic episode.

    ``base_df`` is the original episode's actions.csv; only the perturbed joint
    and EE-position columns are overwritten (orientation & gripper are unchanged).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = base_df.copy()
    n = len(pert.traj)

    # Joints (rad -> 0.001 deg), first 6 DoF.
    j = feas.joints[:, :6] / MDEG_TO_RAD
    for k in range(6):
        df.loc[: n - 1, f"{arm}_j{k + 1}"] = np.round(j[:, k]).astype(np.int64)
    # EE position (m -> 0.001 mm). Orientation columns unchanged (offset is pure
    # translation), so we leave *_rx/ry/rz as recorded.
    pos = pert.traj.positions / MM_001_TO_M
    for c, col in zip(range(3), (f"{arm}_ee_x", f"{arm}_ee_y", f"{arm}_ee_z")):
        if col in df.columns:
            df.loc[: n - 1, col] = np.round(pos[:, c]).astype(np.int64)

    df.to_csv(out / "actions.csv", index=False, lineterminator="\r\n")
    _write_pngs(out / "realsense_color", wrist_frames)
    _write_pngs(out / "zed_color", zed_frames)

    meta = {
        "source_episode": source_episode,
        "label": "success" if feas.success else "failure",
        "grasp_frame": pert.grasp_frame,
        "perturbation": {
            "name": pert.spec.name,
            "base_offset_m": list(pert.spec.base_offset),
            "envelope": pert.spec.envelope,
        },
        "feasibility": {
            "success": bool(feas.success),
            "max_residual": feas.max_residual,
            "n_unreachable": feas.n_unreachable,
            "n_frames": int(n),
        },
        "columns_overwritten": [f"{arm}_j1..j6", f"{arm}_ee_x/y/z"],
    }
    (out / "perturbation.json").write_text(json.dumps(meta, indent=2))
    return out
