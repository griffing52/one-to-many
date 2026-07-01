"""Cheap world model: one recorded episode -> many synthetic episodes.

Perturb a recorded trajectory with a converging offset (reaches the same grasp),
label its IK feasibility (success/failure), and render both the wrist (depth-warp)
and third-person (URDF robot over the ZED clean plate) views, writing the result
back in the input dataset format. See ``docs/worldmodel.md`` and
``configs/worldmodel.yaml``.
"""
from .perturb import (PerturbationSpec, PerturbedTrajectory, check_feasibility,
                      detect_grasp_frame, offset_envelope, perturb_trajectory)
from .pipeline import SyntheticEpisodePipeline, WorldModelConfig
from .synth import write_synthetic_episode
from .thirdperson import ThirdPersonRenderer, load_zed_camera
from .wrist_warp import (GripperMask, WristIntrinsics, WristWarper,
                         base_offset_to_camera, disparity_to_depth)

__all__ = [
    "PerturbationSpec", "PerturbedTrajectory", "perturb_trajectory",
    "check_feasibility", "detect_grasp_frame", "offset_envelope",
    "SyntheticEpisodePipeline", "WorldModelConfig",
    "write_synthetic_episode",
    "ThirdPersonRenderer", "load_zed_camera",
    "GripperMask", "WristIntrinsics", "WristWarper",
    "base_offset_to_camera", "disparity_to_depth",
]
