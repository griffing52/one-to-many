"""End-to-end synthetic-data pipeline for the cheap world model.

Given one recorded episode and a :class:`PerturbationSpec`, this:

  1. loads the recorded EE trajectory + measured joints,
  2. perturbs the trajectory with a converging offset (:mod:`.perturb`),
  3. checks IK feasibility -> success/failure label,
  4. renders the **wrist** view (depth-warp, gripper fixed) and the
     **third-person** view (URDF robot over the ZED clean plate) per frame,
  5. writes a synthetic episode in the input dataset format (:mod:`.synth`).

Everything is driven by ``configs/worldmodel.yaml`` -> :class:`WorldModelConfig`.
This is the module a script (``scripts/07_synthesize_episode.py``) calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..data import Episode, load_ee_trajectory, load_joint_trajectory
from ..utils.logging import get_logger
from .perturb import (PerturbationSpec, PerturbedTrajectory, check_feasibility,
                      perturb_trajectory)
from .synth import write_synthetic_episode
from .thirdperson import ThirdPersonRenderer, load_zed_camera
from .wrist_warp import (GripperMask, WristIntrinsics, WristWarper,
                         base_offset_to_camera, disparity_to_depth)

log = get_logger(__name__)


@dataclass
class WorldModelConfig:
    """Resolved config for the synthetic pipeline (see configs/worldmodel.yaml)."""
    episode_dir: Path
    urdf: Path                     # FK/IK URDF (camera variant, exposes hand_cam)
    render_urdf: Path              # MuJoCo render URDF (STL)
    urdf_dir: Path
    zed_extrinsic_npz: Path
    clean_plate: Path
    wrist_intr: WristIntrinsics
    gripper_mask: GripperMask
    spec: PerturbationSpec
    output_dir: Path
    ee_frame: Optional[str] = None
    camera_frame: str = "hand_cam"
    base_frame: str = "base_link"
    arm: str = "slave"
    ee_source: str = "ee"
    ik_tol: float = 5e-3
    kernel_splat: bool = True
    inpaint_holes: bool = True
    fill_method: str = "inpaint"       # none|nearest|bilinear|edge_aware|inpaint
    wrist_renderer: str = "depthwarp"  # depthwarp | genwarp
    genwarp_mode: str = "pad"          # pad | crop | squash
    genwarp_depth_scale: float = 1.0   # <1 exaggerates the shift, >1 shrinks it
    genwarp_steps: int = 20
    genwarp_guidance: float = 3.5
    frame_range: Optional[Tuple[int, int]] = None
    render_wrist: bool = True
    render_zed: bool = True


class SyntheticEpisodePipeline:
    def __init__(self, cfg: WorldModelConfig):
        self.cfg = cfg
        self._mono = None

    # -- lazy heavy singletons ------------------------------------------------
    @property
    def mono(self):
        if self._mono is None:
            from ..depth import MonoDepthEstimator
            log.info("Loading Depth-Anything-V2 ...")
            self._mono = MonoDepthEstimator()
        return self._mono

    def run(self) -> Path:
        cfg = self.cfg
        ep = Episode(cfg.episode_dir)
        df = ep.actions_df()
        traj = load_ee_trajectory(df, arm=cfg.arm, source=cfg.ee_source)
        meas_joints = load_joint_trajectory(df, arm=cfg.arm)

        # 1-2. perturb ------------------------------------------------------
        pert = perturb_trajectory(traj, cfg.spec)
        log.info("Grasp frame = %d, envelope=%s, offset=%s",
                 pert.grasp_frame, cfg.spec.envelope, cfg.spec.base_offset)

        # 3. feasibility / label -------------------------------------------
        from ..robot import PiperModel
        model = PiperModel(str(cfg.urdf), str(cfg.urdf_dir),
                           base_frame=cfg.base_frame, ee_frame=cfg.ee_frame,
                           camera_frame=cfg.camera_frame)
        feas = check_feasibility(model, pert, meas_joints, tol=cfg.ik_tol)
        log.info("Feasibility: %s (max_residual=%.4f, unreachable=%d/%d)",
                 "SUCCESS" if feas.success else "FAILURE",
                 feas.max_residual, feas.n_unreachable, len(pert.traj))

        # frame range (default: whole episode)
        n = len(pert.traj)
        lo, hi = (0, n) if cfg.frame_range is None else cfg.frame_range
        idx = list(range(lo, hi))

        # wrist-cam FK rotations (base<-hand_cam) at ORIGINAL joints per frame.
        cam_R = [model.fk(meas_joints[i], [cfg.camera_frame])[cfg.camera_frame][:3, :3]
                 for i in idx]

        # 4a. wrist view (depth-warp OR genwarp) ---------------------------
        wrist_frames: List[np.ndarray] = []
        if cfg.render_wrist:
            warper = WristWarper(cfg.wrist_intr, cfg.gripper_mask,
                                 kernel_splat=cfg.kernel_splat,
                                 inpaint_holes=cfg.inpaint_holes,
                                 fill_method=cfg.fill_method)
            gw = None
            if cfg.wrist_renderer == "genwarp":
                from .genwarp_warp import GenWarpWrapper
                gw = GenWarpWrapper(num_inference_steps=cfg.genwarp_steps,
                                    guidance_scale=cfg.genwarp_guidance)
                gmask = cfg.gripper_mask.mask(cfg.wrist_intr.height, cfg.wrist_intr.width)
            wrist_paths = ep.wrist_frames()
            for k, i in enumerate(idx):
                real = np.asarray(Image.open(wrist_paths[i]).convert("RGB"))
                depth = disparity_to_depth(self.mono.estimate(real))
                dcam = base_offset_to_camera(pert.offsets[i], cam_R[k])
                if gw is not None:
                    frame = gw.warp(real, depth, dcam, cfg.wrist_intr.fy,
                                    mode=cfg.genwarp_mode,
                                    depth_scale=cfg.genwarp_depth_scale)
                    frame[gmask] = real[gmask]     # keep the gripper fixed
                    wrist_frames.append(frame)
                else:
                    wrist_frames.append(warper.warp(real, depth, dcam))
                if k % 40 == 0:
                    log.info("  wrist %s %d/%d (|dcam|=%.3fm)", cfg.wrist_renderer,
                             k, len(idx), float(np.linalg.norm(dcam)))

        # 4b. third-person view (robot over clean plate) -------------------
        zed_frames: List[np.ndarray] = []
        if cfg.render_zed:
            from ..robot import RobotRenderer
            renderer = RobotRenderer(str(cfg.render_urdf), str(cfg.urdf_dir))
            cam = load_zed_camera(cfg.zed_extrinsic_npz)
            plate = np.asarray(Image.open(cfg.clean_plate).convert("RGB"))
            tp = ThirdPersonRenderer(renderer, cam, plate)
            for k, i in enumerate(idx):
                zed_frames.append(tp.render(feas.joints[i]))
                if k % 40 == 0:
                    log.info("  zed render %d/%d", k, len(idx))

        # 5. write synthetic episode ---------------------------------------
        # slice the perturbation/feasibility to the rendered range for the CSV.
        sub = _slice(pert, feas, idx)
        out = write_synthetic_episode(
            cfg.output_dir, df.iloc[idx].reset_index(drop=True), sub[0], sub[1],
            wrist_frames or [np.zeros((cfg.wrist_intr.height, cfg.wrist_intr.width, 3), np.uint8)] * len(idx),
            zed_frames or [np.zeros((720, 1280, 3), np.uint8)] * len(idx),
            arm=cfg.arm, source_episode=ep.episode_id)
        log.info("Wrote synthetic episode -> %s (label=%s)", out,
                 "success" if feas.success else "failure")
        return out


def _slice(pert: PerturbedTrajectory, feas, idx: List[int]):
    """Slice trajectory + feasibility to the rendered frame range."""
    from ..data.types import EETrajectory
    from .perturb import FeasibilityReport
    t = pert.traj
    sub_traj = EETrajectory(t.timestamps[idx], t.positions[idx],
                            t.rotvecs[idx], t.gripper[idx])
    sub_pert = PerturbedTrajectory(sub_traj, pert.offsets[idx], pert.weights[idx],
                                   pert.grasp_frame, pert.spec)
    sub_feas = FeasibilityReport(feas.reachable[idx], feas.residuals[idx],
                                 feas.joints[idx], feas.success,
                                 feas.max_residual, feas.n_unreachable)
    return sub_pert, sub_feas
