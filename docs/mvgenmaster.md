# Wrist synthesis — MVGenMaster (multi-view diffusion NVS)

[MVGenMaster](https://github.com/ewrfcas/MVGenMaster) (CVPR 2025) is the third
wrist renderer, next to `depthwarp` and `genwarp`. Where GenWarp generates the
shifted view from **one** source frame, MVGenMaster is a multi-view diffusion
model: it conditions on a **variable number of reference views with known
cameras** plus 3D priors (reference pixels warped by depth into each target),
and generates any number of target views in one pass. That is exactly the shape
of our problem — we have *many* real frames of the same static scene and *know
every camera*:

- **wrist refs**: FK of the `hand_cam` link at the measured joints, remapped to
  the OpenCV optical convention (`wrist_c2w`) — metric, base frame;
- **third-person ref**: the calibrated ZED (`zed_extrinsic.npz`) with the SAM
  clean plate and its **metric** scene depth (`zed_scene_metric.npz`);
- **targets**: the same wrist pose translated by the base-frame perturbation
  offset (`offset_target_c2w`) — identical parallax to the depth-warp's `dcam`.

So unlike the upstream demo (which needs DUSt3R to *estimate* unknown poses),
we feed the model ground-truth cameras and only use DUSt3R for **reference
depth**, locked to the known poses (`preset_pose`/`preset_focal` +
`init="known_poses"`), which returns dense depth **on the metric FK scale** —
the scale consistency the 3D priors require. `ref_depth: vda` reuses the
Video-Depth-Anything depth instead (pseudo-metric, median-0.5 m; tune with
`mvgen.depth_scale`).

## Setup

- Repo cloned at `a2l/MVGenMaster` (path hardcoded in
  `src/o2m/worldmodel/mvgen_warp.py::_MVGEN_REPO`).
- Checkpoints in `a2l/MVGenMaster/check_points/`:
  `pretrained_model/` (config.yaml + ema_unet.pt + ratio_set.json, from the
  repo's HF release zip) and `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`.
- SD-2.1 base (`stabilityai/stable-diffusion-2-1`) in the HF cache — downloaded
  automatically on first load (unet/vae/text-encoder shells; the multi-view
  ema_unet.pt then replaces the UNet weights).
- Runs **in-process in the `o2m` conda env** (torch 2.12 / transformers 5.12 /
  diffusers 0.38): the repo vendors its own `my_diffusers` 0.29, which needed
  only a tiny compat patch (`pipeline_loading_utils.py`: transformers-5 dropped
  the FLAX/SAFE/WEIGHTS name constants — now try/except). NOTE the pinned
  upstream env (torch 2.5.1) cannot drive the RTX 5070 Ti (sm_120) anyway.

## Compare conditioning strategies on one frame

```bash
PYTHONPATH=src python scripts/12_mvgen_demo.py \
    --frame 40 --offset 0 0.04 0.02 --k 5
# -> outputs/episode_000/renders/worldmodel/mvgen_strategies_f40_dust3r.png
```

Strategies: `self` (only frame *t*), `pm_k` (*t±k* + *t*), `pm_2k` (*t±k, t±2k*
+ *t*), `nbrs_only` (*t±k* without *t*), `zed` (*t* + ZED plate), `zed_k`
(*t±k* + *t* + ZED). Baselines in the montage: the real frame and the
depth-warp of the same offset/depth.

## Episode pipeline

```bash
PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py \
    --wrist-renderer mvgen
```

Frames are generated in **chunks** (`mvgen.chunk`, default 6): each diffusion
call conditions on `mvgen.refs` real frames sampled evenly across the chunk
(+ the ZED plate when `mvgen.use_zed_ref`) and synthesizes all the chunk's
perturbed poses at once. Zero-offset frames (after the grasp under the `bump`
envelope) are passed through untouched. The gripper trapezoid is copied from
each real frame, as in the other renderers.

## Knobs (`configs/worldmodel.yaml -> worldmodel.mvgen`)

| Knob | Default | Meaning |
|---|---|---|
| `ref_depth` | `dust3r` | ref-depth source: `dust3r` (metric, per chunk) / `vda` |
| `chunk` | 6 | perturbed frames per diffusion call |
| `refs` | 3 | reference frames (spread/nearest; hybrid global keyframes) |
| `ref_select` | `hybrid` | `hybrid` (global keyframes, SAME every chunk, + per-chunk nearest — best video cohesion) / `nearest` (min-shift refs) / `spread` |
| `near_refs` | 2 | per-chunk nearest refs added in hybrid mode |
| `rot_weight` | 0.1 | m-per-rad rotation term in the nearest-pose distance |
| `min_baseline` | 0.06 | widen refs until the FK camera span reaches this (m) |
| `max_align_loss` | 0.02 | dust3r loss gate; above it -> anchored-VDA fallback |
| `use_zed_ref` | false | add the ZED clean plate as an extra reference view |
| `num_inference_steps` | 50 | DDIM steps |
| `guidance_scale` | 2.0 | CFG (upstream default) |
| `depth_scale` | 1.0 | ref-depth multiplier — only meaningful with `vda` |

Measured on the full `left_up_4_2cm` episode (frame-to-frame flicker, live range;
real video floor 8.9): spread 18.6, nearest 17.9, **hybrid 15.8** — and hybrid
also removed the rack-segment hallucinations both others showed (the constant
global-keyframe conditioning keeps appearance consistent across chunk
boundaries; the per-chunk nearest refs minimise the synthesis shift).

## Caveats

- All views of one call share a single resolution; a ZED ref is centre-cropped
  to the wrist 4:3 aspect first (`crop_to_aspect`, fixes `cx`), then the model
  resizes everything to its `ratio_set` resolution (512×384 for 4:3) — output
  frames are LANCZOS-upscaled back to 640×480, so fine texture is slightly
  softer than depth-warp's native-resolution pixels.
- The gripper is rigidly mounted, so in wrist refs it sits at the SAME pixels in
  every view — geometrically it is a moving object the static-scene model can't
  explain. It sits inside the fixed trapezoid that we overwrite anyway; a
  future refinement is masking it out of the refs' depth.
- The model was trained with ≤3 condition views per scene sample
  (`max_cond_num: 3`); more refs still work (the demo's `pm_2k` uses 5) but
  are outside the training distribution.
