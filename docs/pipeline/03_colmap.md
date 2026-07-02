# 03 — COLMAP poses

!!! note "Splat route only"
    COLMAP poses feed **splat** training and its sim3 alignment. The synthetic-data
    pipeline does not run SfM — it calibrates the fixed ZED into the base frame by
    target-free PnP (`outputs/<ep>/align/zed_extrinsic.npz`) and gets wrist geometry
    from monocular depth. (Feature matching *cannot* connect the ZED to the wrist
    anyway — see `notes.md`.)

`scripts/03_run_colmap.py` → `o2m.colmap`

**In**: `frames/` + `masks/` + `configs/camera.yaml`.
**Out**: `colmap/sparse/0/*`, `nerfstudio/transforms.json`,
`nerfstudio/sparse_pc.ply`, and symlinked `images/` + `masks/`.

Runs feature extraction → sequential matching → sparse mapping with a single
shared `OPENCV` camera, then converts the model to a Nerfstudio
`transforms.json` (see [data contracts](../data_contracts.md)).

```bash
python scripts/03_run_colmap.py --episode episode_000
```

**Quality gates** (`configs/colmap.yaml`): registered fraction and mean
reprojection error. The sparse point cloud seeds splatfacto and feeds the
alignment sanity checks.
