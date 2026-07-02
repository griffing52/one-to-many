# Usage

## Environment

Use the prepared **conda `o2m`** env (torch+CUDA, nerfstudio, gsplat, pycolmap,
mujoco, pinocchio, ninja). Run scripts with that interpreter:

```bash
conda activate o2m
python scripts/run_pipeline.py ...
```

Notes from bring-up:

- **No COLMAP system binary needed** — the runner uses the `pycolmap` Python API
  automatically when the `colmap` CLI is absent (`colmap.backend_impl: auto`).
- **gsplat compiles CUDA kernels on first run** and needs the `ninja` binary on
  PATH. The trainer prepends the interpreter's bin dir and sets `CUDA_HOME`
  automatically, so `conda activate` is enough; first training run is slow while
  kernels build.
- **OOM during first build (important on shared boxes)**: gsplat compiles many
  kernels and each `cicc` (CUDA compiler) uses ~3 GB RAM; compiling them in
  parallel exhausted this 29 GB machine and the OOM killer (earlyoom) killed the
  build mid-compile. The trainer now sets `MAX_JOBS=2` to cap parallel compiles
  (~6 GB peak). Kernels are cached after the first successful build, so the cost
  is paid once. Lower to `MAX_JOBS=1` if a co-running job is also eating RAM.
- **Headless rendering**: export `MUJOCO_GL=egl` before stage 05 (no display).
- The `.venv/` created by `uv sync` is a separate, incomplete env — ignore it.

## Synthetic-data pipeline (current — no splat)

The main path needs **no COLMAP and no splat training** — just the mono-depth model,
the persisted ZED→base calibration, and the clean plate (both already in
`outputs/episode_000/`). Smoke test on a short frame range:

```bash
# 20 frames around the grasp, both views, depth-warp wrist renderer:
PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py \
    --frames 120 141 --offset 0 0.06 0.03 --name smoke_left_up

# compare methods on one frame (writes a side-by-side PNG):
PYTHONPATH=src MUJOCO_GL=egl python scripts/09_fill_methods_demo.py --frame 130

# original-vs-synthetic montage video:
PYTHONPATH=src python scripts/08_compare_video.py \
    --synthetic outputs/episode_000/synthetic/smoke_left_up \
    --raw /home/.../pick_bag_joe/episode_000
```

Full guide (all knobs, GenWarp, sweeps): **[Generate synthetic episodes](synthetic_data.md)**.
`MUJOCO_GL=egl` is required (the third-person view renders the URDF arm headless);
GenWarp additionally needs its checkpoints — see [wrist synthesis](genwarp.md).

## Legacy splat pipeline (free-viewpoint reconstruction)

Only needed to train a splat for flythroughs / `robot_overlay`.

### End-to-end (one episode, smoke settings)

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml \
    --episode episode_000 --max-iters 2000
```

### Stage by stage

```bash
python scripts/01_extract_frames.py    --episode episode_000 --frame-stride 4
# Stage 02 (masking) is OFF by default — the bundled motion masker is wrong for a
# moving camera. Skip it, or use a learned masker (see pipeline/02_masking).
python scripts/03_run_colmap.py        --episode episode_000   # wrist-only poses
# ...OR, to also register the third-person ZED demo viewpoint (supersedes 03):
python scripts/03b_register_zed.py     --episode episode_000
python scripts/04b_align_world.py      --episode episode_000   # splat<->base sim3
python scripts/04_train_splat.py       --episode episode_000 --max-iters 2000
MUJOCO_GL=egl python scripts/05_render_trajectory.py --episode episode_000 \
    --mode robot_overlay --edit replay_original
```

The layers degrade gracefully: skip 03b → render falls back to a wrist
viewpoint; skip 04b → only `--mode static` is available (no robot overlay).

Artifacts land under `outputs/<episode>/` (see `o2m.config.EpisodePaths`).

### Verification recipe (splat)

1. **Masks** (after stage 02): eyeball a few `masks/*.png` — arm/bag black, table
   white.
2. **COLMAP** (stage 03): > 85 % images registered, mean reproj error ≲ 1.5 px,
   sparse cloud is background only. (`notebooks/verify_alignment.py` plots the
   camera path.)
3. **Alignment** (stage 04b): `residual_rms` ≪ table size (target < 5 cm), scale
   physically plausible.
4. **Splat** (stage 04, smoke): training views are sharp with no arm ghost.
5. **Static render**: `--mode static` flies a clean orbit.
6. **Decisive check**: `--mode robot_overlay --edit replay_original` and overlay
   the result on the original ZED video — the rendered arm should land on the
   real arm.
7. Only then try an altered edit (`--edit shift_xy`), and finally a full-quality
   run (`--max-iters 30000`).

### Trajectory edits (splat)

Presets live in `o2m.trajectory.presets` (`replay_original`, `shift_xy`,
`higher_lift`, `wider_reach`, `slow_motion`). Add your own there or build them
with `TrajectoryEditor`.
