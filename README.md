# one-to-many

Turn **one** recorded robot demonstration into **many**: reconstruct the
environment as a 3D Gaussian Splat and re-render the scene under *altered*
end-effector (EE) trajectories.

The pipeline takes AgileX/Piper teleop episodes (a moving wrist camera + a
stationary third-person camera + per-frame EE/joint logs), reconstructs the
static environment with [Nerfstudio](https://docs.nerf.studio/) `splatfacto`
(camera poses from COLMAP, dynamic foreground masked out), then renders new
trajectories in three modes:

- **static** — fly a free/orbit camera through the environment splat;
- **robot_overlay** *(main)* — render the Piper arm from its URDF at edited EE
  poses and composite it into splat renders from the original camera viewpoint;
- **dynamic** *(stretch)* — additionally re-pose the manipulated object.

## Installation

```bash
# Core (data + kinematics + compositing)
uv pip install -e .

# Reconstruction stack (CUDA-specific; pin in your environment)
uv pip install -e ".[splat,robot]"
```

## Quickstart

```bash
# End-to-end on one episode (smoke settings)
python scripts/run_pipeline.py --config configs/pipeline.yaml --episode episode_000

# Or run stages individually
python scripts/01_extract_frames.py   --config configs/pipeline.yaml --episode episode_000
python scripts/02_generate_masks.py   --config configs/pipeline.yaml --episode episode_000 --masker color
python scripts/03_run_colmap.py       --config configs/pipeline.yaml --episode episode_000
python scripts/04_train_splat.py      --config configs/pipeline.yaml --episode episode_000
python scripts/05_render_trajectory.py --config configs/pipeline.yaml --episode episode_000 \
    --mode robot_overlay --edit replay_original
```

## Documentation

```bash
mkdocs serve   # browse docs/ locally
```

See `docs/` for the data format, the COLMAP / Nerfstudio data contracts, the
splat-world ↔ robot-base alignment (the crux of the overlay), and per-stage
usage.
