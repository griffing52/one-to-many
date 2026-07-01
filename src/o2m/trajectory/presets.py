"""Named trajectory edits referenced from configs/pipeline.yaml (render.edit)."""
from __future__ import annotations

from typing import Callable, Dict

from ..data.types import EETrajectory
from .editor import TrajectoryEditor

PRESETS: Dict[str, Callable[[EETrajectory], EETrajectory]] = {
    "replay_original": lambda t: t.copy(),
    "shift_xy": lambda t: TrajectoryEditor(t).translate((0.05, 0.05, 0.0)).result(),
    "higher_lift": lambda t: TrajectoryEditor(t).lift(0.05).result(),
    "wider_reach": lambda t: TrajectoryEditor(t).scale_about_start(1.2).result(),
    "slow_motion": lambda t: TrajectoryEditor(t).time_warp(1.5).result(),
}


def apply_preset(name: str, traj: EETrajectory) -> EETrajectory:
    if name not in PRESETS:
        raise KeyError(f"Unknown trajectory preset {name!r}. Options: {list(PRESETS)}")
    return PRESETS[name](traj)
