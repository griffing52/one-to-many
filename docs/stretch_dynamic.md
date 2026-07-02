# Real2Sim: scene + manipulated object (compositional)

`robot_overlay` reconstructs only the **static** environment from the wrist
trajectory. The manipulated object (the bag) is the whole point of the demo, but
splatting the *full sequence* smears it — once the robot lifts the bag the scene
is no longer static, so the gaussians after pickup go wild.

The fix is **not** a 4D / time-varying splat. It's **compositional real2sim**:
reconstruct the scene *once while it is static*, then animate the robot and the
object analytically on top.

## The static moment

The stationary ZED third-person view is static at the **start** (robot down, bag
on the rack) and again at the **end** (bag placed on the table); it is dynamic in
between. So **ZED frame 0 is the reference static scene** (verified:
`renders/real2sim/zed_survey.png`).

## Validated building blocks (2026-06-30)

1. **Static "stage" from one frame** — Depth-Anything-V2 on ZED f0 → a coherent
   2.5D scene (table, rack, both bags, gripper, wall) that holds up under
   viewpoint rotation (`renders/real2sim/r2s_scene_views.png`). Because the demo
   viewpoint *is* the ZED, a single-view 2.5D is enough — no missing-backside
   problem as long as we render from near the capture vantage.
2. **Object isolation without SAM** — GrabCut seeded by a box cleanly extracts
   the bag(s) from f0 (`renders/real2sim/r2s_bag_segment.png`). Each becomes a
   movable layer with its own DA-v2 depth patch.
3. **Robot layer** — URDF + Pinocchio FK rendered via MuJoCo (the `robot`
   module), to be drawn from the ZED viewpoint.

## The pipeline

```
ZED f0 (static) ─▶ DA-v2 depth ─▶ static stage (2.5D, background)
              └─▶ segment bag(s) ─▶ movable object layer(s) + depth
robot URDF + FK ─▶ robot layer (rendered from ZED viewpoint)
                         │
   altered EE trajectory ┤  bag rigidly attached to gripper after grasp:
                         │     T_obj(t) = T_ee(t) · T_ee←obj  (fit once at grasp)
                         ▼
   composite [stage ◁ bag ◁ robot] from the ZED viewpoint, depth-ordered
```

This never splats the dynamic sequence, so there are no post-pickup artifacts.

## What's needed to close the loop

- ~~**ZED ↔ robot-base calibration**~~ **(done)** — target-free PnP on the tracked
  green gripper (~7 px) places the URDF robot into the ZED base frame; persisted at
  `outputs/<ep>/align/zed_extrinsic.npz` and already used by the synthetic-data
  pipeline's third-person view.
- ~~**Clean plate**~~ **(done)** — the robot is SAM-segmented and **inpainted out**
  of ZED f0 (`clean_plate_sam.png`), the background the arm composites over.
- **Grasp moment** (when the gripper closes on the bag) → attach the bag to the
  EE and fit `T_ee←obj` so it follows the gripper under perturbation. *(open)*
- **Compositing**: depth-ordered alpha-over; **inpaint** the rack spot the bag
  vacates (for "pick from rack"; minor since it's a small region). *(open)*

The static-scene calibration + clean-plate pieces are now wired into
`o2m.worldmodel` (see [world model](worldmodel.md)); what remains for *dynamic
objects* is the grasp-time bag attach and the re-posed-object compositing/inpaint.

## Approach options

| Approach | Stage | Object | Best for |
|----------|-------|--------|----------|
| **A — 2.5D compositional** (recommended first) | ZED f0 DA-v2 2.5D | GrabCut/SAM layer + depth | render from the fixed third-person demo view; simplest, robust |
| **B — static multi-view splat** | 3DGS from pre-pickup frames | separately reconstructed bag splat | free-viewpoint, photoreal; limited by low-texture SfM |
| **C — hybrid** | DA-v2 2.5D background | analytic object (mesh/splat) + URDF robot | balance of robustness and quality |
