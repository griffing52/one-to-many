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
                         base_offset_to_camera, disparities_to_depths,
                         disparity_to_depth)

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
    depth_estimator: str = "video"     # video (Video-DA, consistent) | mono (DA-v2/frame)
    depth_encoder: str = "vits"        # vits | vitb | vitl (video path only)
    wrist_renderer: str = "depthwarp"  # depthwarp | genwarp | mvgen
    genwarp_mode: str = "pad"          # pad | crop | squash
    genwarp_depth_scale: float = 1.0   # <1 exaggerates the shift, >1 shrinks it
    genwarp_steps: int = 20
    genwarp_guidance: float = 3.5
    # MVGenMaster (wrist_renderer: mvgen) — multi-view diffusion NVS, chunked.
    mvgen_ref_depth: str = "dust3r"    # dust3r (metric, per chunk) | vda
    mvgen_chunk: int = 6               # perturbed frames generated per call
    mvgen_refs: int = 3                # real frames used as refs per chunk
    mvgen_use_zed: bool = False        # add the ZED clean plate as an extra ref
    mvgen_steps: int = 50
    mvgen_guidance: float = 2.0
    mvgen_depth_scale: float = 1.0     # ref-depth multiplier (vda depth only)
    # dust3r ref-depth conditioning: refs are widened beyond the chunk until the
    # FK camera baseline reaches min_baseline (tiny/static baselines + the FK
    # mount-rotation error break the preset-pose solve); a chunk whose alignment
    # loss still exceeds max_align_loss falls back to VDA depth rescaled to the
    # last well-aligned chunk's metric median.
    # 6cm ~ the strategy demo's known-good ref span; 3cm gave borderline depth
    # (loss ~0.01) whose chunks hallucinated. Healthy chunks converge ~0.001-0.01.
    mvgen_min_baseline: float = 0.06
    mvgen_max_align_loss: float = 0.02
    # spread : refs = evenly spaced frames across the baseline-widened chunk.
    # nearest: refs = real frames with poses closest to the perturbed targets
    #          (minimal synthesis shift; depth solved with spread aux frames).
    # hybrid : GLOBAL keyframes spread over the whole live trajectory (the SAME
    #          for every chunk -> consistent appearance across chunk boundaries)
    #          + per-chunk nearest refs (minimal shift). Best video cohesion.
    mvgen_ref_select: str = "hybrid"
    mvgen_rot_weight: float = 0.1      # metres-per-radian in the pose distance
    mvgen_near_refs: int = 2           # per-chunk nearest refs (hybrid mode)
    frame_range: Optional[Tuple[int, int]] = None
    render_wrist: bool = True
    render_zed: bool = True
    # Metric plate depth (scripts/10_zed_metric_scene.py) -> depth-ordered
    # third-person composite (arm hides behind nearer scene geometry).
    scene_depth_npz: Optional[Path] = None
    depth_margin: float = 0.12


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

    def _estimate_depths(self, reals: List[np.ndarray]) -> List[np.ndarray]:
        """Wrist depth for every frame. ``video`` (default) runs
        Video-Depth-Anything over the whole sequence -> temporally consistent
        depth, normalised ONCE globally; ``mono`` is the old per-frame DA-v2
        path (per-frame normalisation, flickers on video)."""
        if self.cfg.depth_estimator == "video":
            from ..depth import VideoDepthEstimator
            log.info("Loading Video-Depth-Anything (%s) ...", self.cfg.depth_encoder)
            vde = VideoDepthEstimator(encoder=self.cfg.depth_encoder)
            return list(disparities_to_depths(vde.estimate_sequence(reals)))
        return [disparity_to_depth(self.mono.estimate(r)) for r in reals]

    def _render_wrist_mvgen(self, pert, idx: List[int], reals: List[np.ndarray],
                            cam_T: List[np.ndarray]) -> List[np.ndarray]:
        """Wrist view via MVGenMaster: chunks of perturbed target poses generated
        from a few REAL frames of the same chunk (+ optionally the ZED plate) as
        reference views with known FK cameras. Zero-offset frames are passed
        through untouched (no resampling of already-real pixels)."""
        cfg = self.cfg
        from .mvgen_warp import (MVGenWrapper, crop_to_aspect, offset_target_c2w,
                                 wrist_c2w)
        wi = cfg.wrist_intr
        K = np.array([[wi.fx, 0, wi.cx], [0, wi.fy, wi.cy], [0, 0, 1.0]])
        gmask = cfg.gripper_mask.mask(wi.height, wi.width)
        c2ws = [wrist_c2w(T) for T in cam_T]
        mv = MVGenWrapper(num_inference_steps=cfg.mvgen_steps,
                          guidance_scale=cfg.mvgen_guidance)

        zed_ref = None
        if cfg.mvgen_use_zed:
            from .scene_cloud import load_scene_depth
            zdep = load_scene_depth(cfg.scene_depth_npz) if cfg.scene_depth_npz else None
            if zdep is None:
                log.warning("mvgen_use_zed: no metric plate depth -> ZED ref OFF")
            else:
                z = np.load(str(cfg.zed_extrinsic_npz))
                plate = np.asarray(Image.open(cfg.clean_plate).convert("RGB"))
                zed_ref = crop_to_aspect(plate, z["K"], wi.height / wi.width, depth=zdep)
                zed_ref = (*zed_ref, z["c2w"])   # (rgb, K, depth, c2w)

        vda_depths = None
        if cfg.mvgen_ref_depth != "dust3r":
            vda_depths = [d * cfg.mvgen_depth_scale for d in self._estimate_depths(reals)]

        out: List[Optional[np.ndarray]] = [None] * len(idx)
        # zero-offset frames (converged / after grasp) stay real
        live = [k for k, i in enumerate(idx)
                if float(np.linalg.norm(pert.offsets[i])) > 1e-6]
        for k in range(len(idx)):
            if k not in set(live):
                out[k] = reals[k]
        good_med = None       # metric median depth of the last well-aligned chunk
        gsel: List[int] = []
        if cfg.mvgen_ref_select == "hybrid" and live:
            # Global keyframes spread over the WHOLE live trajectory: identical in
            # every chunk, so the appearance anchor never changes across chunk
            # boundaries (their wide span also conditions the depth solves).
            gsel = _spread_refs([live[0], live[-1]], c2ws, len(idx),
                                cfg.mvgen_refs, cfg.mvgen_min_baseline)
        for lo in range(0, len(live), cfg.mvgen_chunk):
            chunk = live[lo:lo + cfg.mvgen_chunk]
            tars = [offset_target_c2w(c2ws[k], pert.offsets[idx[k]]) for k in chunk]
            if cfg.mvgen_ref_select == "hybrid":
                near = _nearest_refs(tars, c2ws, cfg.mvgen_near_refs,
                                     cfg.mvgen_rot_weight)
                rsel = sorted(set(gsel) | set(near))
            elif cfg.mvgen_ref_select == "nearest":
                # Real frames whose wrist pose is CLOSEST to the perturbed targets
                # -> the least novel-view synthesis the model has to do.
                rsel = _nearest_refs(tars, c2ws, cfg.mvgen_refs,
                                     cfg.mvgen_rot_weight)
            else:
                rsel = _spread_refs(chunk, c2ws, len(idx), cfg.mvgen_refs,
                                    cfg.mvgen_min_baseline)
            ref_rgbs = [reals[k] for k in rsel]
            ref_c2ws = [c2ws[k] for k in rsel]
            ref_Ks = [K] * len(rsel)
            ref_deps = None
            if cfg.mvgen_ref_depth == "dust3r" and len(rsel) > 1:
                # Nearest refs may be nearly co-located, which is fine for the
                # SYNTHESIS but ill-conditions the preset-pose depth solve — so
                # solve with spread AUX frames included and keep the ref depths.
                # (hybrid already contains the wide-span global keyframes.)
                aux = _spread_refs(chunk, c2ws, len(idx), cfg.mvgen_refs,
                                   cfg.mvgen_min_baseline) \
                    if cfg.mvgen_ref_select == "nearest" else []
                solve = sorted(set(rsel) | set(aux))
                deps_all, align_loss = mv.dust3r_depths(
                    [reals[k] for k in solve], [c2ws[k] for k in solve],
                    [K] * len(solve), return_loss=True)
                if align_loss > cfg.mvgen_max_align_loss:
                    log.warning("  chunk %d-%d dust3r align loss %.3f > %.3f -> "
                                "VDA-depth fallback", chunk[0], chunk[-1],
                                align_loss, cfg.mvgen_max_align_loss)
                    ref_deps = None
                else:
                    ref_deps = [deps_all[solve.index(k)] for k in rsel]
                    good_med = float(np.median(np.stack(deps_all)))
            if ref_deps is None:
                if vda_depths is None:
                    vda_depths = [d * cfg.mvgen_depth_scale
                                  for d in self._estimate_depths(reals)]
                ref_deps = [vda_depths[k] for k in rsel]
                if good_med is not None:   # re-anchor pseudo depth to metric scale
                    s = good_med / max(1e-6, float(np.median(np.stack(ref_deps))))
                    ref_deps = [d * s for d in ref_deps]
            if zed_ref is not None:
                ref_rgbs = ref_rgbs + [zed_ref[0]]
                ref_Ks = ref_Ks + [zed_ref[1]]
                ref_deps = ref_deps + [zed_ref[2]]
                ref_c2ws = ref_c2ws + [zed_ref[3]]
            frames = mv.synthesize(ref_rgbs, ref_deps, ref_c2ws, ref_Ks, tars,
                                   tar_Ks=[K] * len(tars),
                                   out_size=(wi.height, wi.width))
            for k, f in zip(chunk, frames):
                f = f.copy()
                f[gmask] = reals[k][gmask]     # gripper is rigid to the camera
                out[k] = f
            log.info("  wrist mvgen chunk %d-%d (%d refs%s)", chunk[0], chunk[-1],
                     len(ref_rgbs), "+zed" if zed_ref is not None else "")
        return [f if f is not None else reals[k] for k, f in enumerate(out)]

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

        # wrist-cam FK poses (base<-hand_cam link, 4x4) at ORIGINAL joints per frame.
        cam_T = [model.fk(meas_joints[i], [cfg.camera_frame])[cfg.camera_frame]
                 for i in idx]
        cam_R = [T[:3, :3] for T in cam_T]

        # 4a. wrist view (depth-warp, genwarp OR mvgen) ---------------------
        wrist_frames: List[np.ndarray] = []
        if cfg.render_wrist and cfg.wrist_renderer == "mvgen":
            wrist_paths = ep.wrist_frames()
            reals = [np.asarray(Image.open(wrist_paths[i]).convert("RGB")) for i in idx]
            wrist_frames = self._render_wrist_mvgen(pert, idx, reals, cam_T)
        elif cfg.render_wrist:
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
            reals = [np.asarray(Image.open(wrist_paths[i]).convert("RGB")) for i in idx]
            depths = self._estimate_depths(reals)
            for k, i in enumerate(idx):
                real, depth = reals[k], depths[k]
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
            scene_depth = None
            if cfg.scene_depth_npz is not None:
                from .scene_cloud import load_scene_depth
                scene_depth = load_scene_depth(cfg.scene_depth_npz)
                log.info("Depth-ordered third-person composite: %s",
                         "ON" if scene_depth is not None else
                         f"npz missing ({cfg.scene_depth_npz}) -> plain overlay")
            tp = ThirdPersonRenderer(renderer, cam, plate,
                                     scene_depth=scene_depth,
                                     depth_margin=cfg.depth_margin)
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


def _nearest_refs(tar_c2ws: List[np.ndarray], c2ws: List[np.ndarray], want: int,
                  rot_weight: float) -> List[int]:
    """Pick the ``want`` real frames whose wrist pose is closest to ANY of the
    chunk's perturbed target poses (positional distance + ``rot_weight`` [m/rad]
    times the rotation angle) — the refs that minimise how far the diffusion
    model must move each view. Usually that includes each target's own source
    frame (distance = its offset) plus temporal neighbours on the near side."""
    best = np.full(len(c2ws), np.inf)
    for T in tar_c2ws:
        for j, C in enumerate(c2ws):
            dt = float(np.linalg.norm(T[:3, 3] - C[:3, 3]))
            cosang = (np.trace(T[:3, :3].T @ C[:3, :3]) - 1.0) / 2.0
            ang = float(np.arccos(np.clip(cosang, -1.0, 1.0)))
            best[j] = min(best[j], dt + rot_weight * ang)
    return sorted(np.argsort(best)[:want].tolist())


def _spread_refs(chunk: List[int], c2ws: List[np.ndarray], n: int, want: int,
                 min_baseline: float) -> List[int]:
    """Pick ``want`` evenly-spaced ref frames covering the chunk, widening the
    window beyond it until the FK camera baseline reaches ``min_baseline`` (m).
    Sub-centimetre baselines (slow/static segments) make the preset-pose DUSt3R
    solve ill-conditioned — the known FK mount-rotation error dominates."""
    lo, hi = chunk[0], chunk[-1]
    while True:
        sel = sorted({int(round(f)) for f in
                      np.linspace(lo, hi, min(want, hi - lo + 1))})
        pts = np.stack([c2ws[k][:3, 3] for k in sel])
        baseline = max(float(np.linalg.norm(a - b)) for a in pts for b in pts)
        if baseline >= min_baseline or (lo == 0 and hi == n - 1):
            return sel
        lo, hi = max(0, lo - 3), min(n - 1, hi + 3)


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
