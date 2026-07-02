# Synthetic-data pipeline (cheap world model)

Turn **one recorded episode** into a **perturbed, IK-verified, dual-view synthetic
episode** written back in the *exact input dataset format*. This is the drop-in
"synthetic data pipeline": the output folder looks like a real
`agilex_data_collection/pick_bag_joe/episode_NNN`, so the LeRobot converter and
any downstream training treat it identically.

```
recorded episode ──▶ perturb trajectory ──▶ IK feasibility (success/failure)
                                              │
                        ┌─────────────────────┴─────────────────────┐
                   wrist view                                 third-person view
        (mono-depth point cloud of the real          (URDF robot @ IK joints over
         frame → reproject → inpaint holes,            the inpainted ZED clean plate)
         gripper kept fixed; or GenWarp)
                        └─────────────────────┬─────────────────────┘
                                              ▼
                    synthetic episode: actions.csv + realsense_color/ + zed_color/
```

No Gaussian splat is trained: the **wrist** background is the real frame lifted to a
monocular-depth (DA-v2) **point cloud** and reprojected, with disocclusions
**inpainted**; the **third-person** background is the real ZED frame with the robot
**inpainted out** (the "clean plate"). See [Representations](representations.md) and
[Wrist synthesis & GenWarp](genwarp.md) for the geometry and fill methods.

## TL;DR — run it

```bash
cd a2l/one-to-many
# Full episode, both views, using the offset in the config:
PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py

# Quick test: 20 frames, custom offset & name:
PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py \
    --frames 110 130 --offset 0 0.06 0.03 --name left_up_6_3cm
```

Output → `outputs/episode_000/synthetic/<name>/` containing `actions.csv`,
`realsense_color/`, `zed_color/`, `perturbation.json`.

## Where everything lives (edit these)

| What | File |
|------|------|
| **All knobs (offset, envelope, grasp frame, masks, paths)** | `configs/worldmodel.yaml` |
| CLI driver (overrides: `--offset --envelope --grasp-frame --frames --name`) | `scripts/07_synthesize_episode.py` |
| Perturbation + converging envelope + IK feasibility label | `src/o2m/worldmodel/perturb.py` |
| Wrist depth-warp + **gripper trapezoid** + optical remap | `src/o2m/worldmodel/wrist_warp.py` |
| Third-person robot-over-plate render | `src/o2m/worldmodel/thirdperson.py` |
| Synthetic-episode writer (CSV/PNG format) | `src/o2m/worldmodel/synth.py` |
| Orchestrator | `src/o2m/worldmodel/pipeline.py` |
| ZED→base calibration (persisted) | `outputs/episode_000/align/zed_extrinsic.npz` |
| ZED clean plate (robot inpainted out) | `outputs/episode_000/renders/worldmodel/clean_plate_sam.png` |
| Robot URDFs / frames | `configs/robot.yaml` |

## The knobs (`configs/worldmodel.yaml → worldmodel:`)

### Perturbation (`perturb:`)
- **`base_offset: [dx, dy, dz]`** — peak EE offset in metres, robot **base frame**
  (`+y` = robot-left, `+z` = up). This is the main dial.
- **`envelope`** —
  - `bump` *(default, recommended)*: 0 at t=0, smoothly up to the peak mid-approach,
    back to 0 at the grasp frame, 0 after. Both the original and perturbed runs
    **start identically** (home pose) and reach the **same grasp** — only the
    mid-reach deviates. Use this when you want them to "start the same, then imagine
    shifts from there."
  - `converge_at_grasp`: **full** offset at t=0, smoothly →0 at the grasp frame, 0
    after. The perturbed run starts *displaced* and converges to the same grasp.
  - `constant`: rigid shift for the whole episode (pure failure simulation).
- **`grasp_frame: null`** — auto-detected from the gripper-closing edge; set an int
  to force it.
- **`ik_tol`** — success threshold on the IK residual (m/rad). If any frame exceeds
  it the episode is labelled **`failure`** (the a2l-pr "too far to grasp" signal).

### Gripper fixed-mask trapezoid (`gripper_mask:`)
The wrist gripper is rigidly mounted to the camera, so it must **not** move when the
scene shifts — its pixels are copied straight from the original frame inside a
bottom-centre trapezoid.
- `y_start_frac` — where the trapezoid starts (fraction down the image).
- `half_top` — half-width at the top edge (fraction of image width).
- `half_grow` — extra half-width added by the bottom edge.

> **Width was halved** per request: `half_top 0.10 → 0.05`, `half_grow 0.35 → 0.175`.
> To make it wider/narrower again, scale these two numbers.

### Wrist renderer + warp quality (`wrist_renderer`, `warp:`)
- `wrist_renderer` — `depthwarp` (fast, reprojects real pixels) or `genwarp`
  (Sony GenWarp diffusion novel-view; cleanest disocclusions, ~14 s/frame). See
  **[genwarp.md](genwarp.md)**.
- `kernel_splat` — 2×2 forward splat (fewer pin-holes / less spray).
- `inpaint_holes` — master switch for hole filling.
- `fill_method` — disocclusion fill for the depth-warp path: `none | nearest |
  bilinear | edge_aware | inpaint`. Compare all of them (plus GenWarp) on one
  frame with `scripts/09_fill_methods_demo.py`.

### Intrinsics / calibration / IO
- `wrist_intrinsics` — RealSense pinhole (`fx=fy=465, 640×480`).
- `zed_extrinsic_npz`, `clean_plate` — third-person calibration + background.
- `render_wrist` / `render_zed` — toggle either view.
- `frame_range: null` — `[start, end)` to synthesise a sub-range; null = full episode.
- `output_root` — synthetic episodes are written here, one subdir per perturbation.

## Output format

`actions.csv` keeps the **exact original columns and CRLF line endings**; only the
perturbed columns are overwritten:
- `slave_j1..j6` ← IK joints (rad → 0.001 deg), anchored to FK(measured)+offset so a
  zero offset returns the measured joints exactly (arm matches the real robot).
- `slave_ee_x/y/z` ← recorded position + base-frame offset (0.001 mm units).
- Orientation (`*_rx/ry/rz`), gripper, master, and `slave_fk_*` are unchanged.

`perturbation.json` records the spec, grasp frame, and feasibility (`success` /
`failure`, `max_residual`, `n_unreachable`) so every synthetic episode is traceable.

## Generating many variations (augmentation / failure sweep)

Loop the driver over offsets; reachable ones are labelled `success` (valid
augmentation), unreachable ones `failure` (recovery targets for a2l-pr):

```bash
for off in "0 0.02 0.01" "0 0.04 0.02" "0 0.08 0.04" "0 0.12 0.06"; do
  read dy_x dy_y dy_z <<< "$off"
  PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py \
      --offset $dy_x $dy_y $dy_z --name "sweep_${dy_y}_${dy_z}"
done
```

## Inspecting & comparing results

Two helper scripts let you *see* what a run produced and compare strategies:

**`scripts/08_compare_video.py`** — original-vs-synthetic 2×2 montage video
(`[wrist orig | wrist synth]` over `[zed orig | zed synth]`), labelled with the
perturbation name and `success`/`failure`. Run it on any synthetic episode:

```bash
PYTHONPATH=src python scripts/08_compare_video.py \
    --synthetic outputs/episode_000/synthetic/left_up_4_2cm \
    --raw /home/.../pick_bag_joe/episode_000
# -> outputs/episode_000/synthetic/left_up_4_2cm/compare.mp4
```

**`scripts/09_fill_methods_demo.py`** — the method-comparison image: warps one wrist
frame by a chosen offset and tiles **every** hole-fill strategy side-by-side
(`original | none | nearest | bilinear | edge_aware | inpaint`) plus, if the
checkpoints are present, **GenWarp** (`warped` + `synthesized`). Each panel is
timed, and it prints the disocclusion hole percentage. Use it to pick a
`fill_method` / `wrist_renderer`:

```bash
PYTHONPATH=src MUJOCO_GL=egl python scripts/09_fill_methods_demo.py \
    --frame 40 --offset 0 0.08 0.04
# -> outputs/episode_000/renders/worldmodel/fill_methods_f40.png
# flags: --no-genwarp, --genwarp-mode {pad,crop,squash}, --depth-scale, --out
```

## Timing

Full 369-frame trajectory, both views, `depthwarp` wrist renderer: **~145 s**
(~2.4 min) each on this machine (see `outputs/episode_000/synthetic/TIMING.txt`).
The `genwarp` wrist renderer is GPU-bound at **~1.1 s/frame** on an RTX 5070 Ti, so a
full dual-view episode is **~9 min** (measured: ~11 s model load + ~6.6 min wrist +
~6 s ZED + ~1.9 min PNG write). It is *much* slower on weak GPUs (~14–18 s/frame),
where you'd restrict it to showcase clips / key frames.

## The third-person arm (colour + alignment)

The arm over the ZED plate is a MuJoCo render of `robot.render_urdf`. That now points
at a **coloured** model, `piper_description_color.xml` — an MJCF using the STL meshes
(correct, connected geometry) with a per-link Piper colour scheme (white segments,
black base/wrist/gripper) applied as geom `rgba`. Regenerate it with
`scripts/build_color_robot.py`; fall back to the flat-grey STL URDF by pointing
`render_urdf` back at `piper_description.urdf`.

> The OBJ+MTL meshes under `meshes/obj/` are **not** used: they aren't in the
> per-link URDF frames (some don't match the URDF's link decomposition), so mapping
> them onto the link bodies makes the arm disjoint. Finer within-link colour would
> need correctly-framed per-link OBJs.

**Alignment caveat.** The FK/joints are correct (the rendered arm *shape* matches the
recording), but the ZED extrinsic + assumed `fx=700` were calibrated on the green
gripper mostly during the grasp/transport phase, so the arm registers well there
(~f180+) and drifts in the **early approach** / near-field. With `envelope: bump`
the perturbation is zero at the start and grasp, so any start mismatch you see is
this calibration residual, not the shift. The proper fix is to re-solve the ZED
calibration (free `fx`, correspondences across the whole episode) — a known TODO.

## Known limitation

Wrist disocclusion "spray" on thin foreground objects (rack legs, bag handles)
grows past ~5–8 cm of offset. `fill_method` and kernel-splat reduce it; **GenWarp**
(`wrist_renderer: genwarp`) fills it cleanly with generated content. The other real
fix is multi-frame fusion (fill holes from neighbouring wrist frames). See `notes.md`.
