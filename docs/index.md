# one-to-many

Turn **one** recorded robot demonstration into **many** by perturbing its
trajectory and re-rendering both camera views photorealistically — a *cheap world
model*. The current pipeline reconstructs geometry as **monocular-depth point
clouds** and fills what the shift reveals with **inpainting** (classical or
diffusion), rather than training a Gaussian splat of the whole scene.

## Why

We have AgileX/Piper teleoperation episodes: a moving **wrist** camera (RealSense),
a stationary **third-person** camera (ZED), and per-frame EE/joint logs — but no
camera calibration and no depth. To synthesise new trajectories *photorealistically*
(not only in simulation), we:

1. **perturb** the recorded EE path (base-frame offset, converging to the same grasp),
2. label it **feasible/infeasible** with IK (augmentation vs. failure signal),
3. **re-render** the wrist view by lifting the real frame to a point cloud with
   monocular depth and reprojecting it under the shift, filling disocclusions by
   inpainting, and
4. **re-render** the third-person view by compositing a URDF arm (at the IK joints)
   over an **inpainted clean plate** (the scene with the robot painted out).

The output is written back in the **exact input dataset format**, so it is a drop-in
for the LeRobot converter and downstream training.

## The two pipelines

| | **Synthetic-data pipeline** *(current / main)* | **Splat reconstruction** *(legacy / free-viewpoint)* |
|---|---|---|
| Geometry | per-frame **mono-depth point cloud** of the real frame | one trained **3D Gaussian splat** of the static scene |
| Wrist view | depth-warp reproject + **inpaint** holes, or **GenWarp** diffusion | render the splat from the moving path |
| 3rd-person | URDF arm over an **inpainted clean plate** | splat + composited URDF arm (`robot_overlay`) |
| Needs | mono-depth, ZED→base calibration, clean plate | COLMAP poses + splat training (slow, GPU) |
| Best for | small–moderate perturbations near the real path; failure sim | free orbit / flythrough of the static environment |
| Package | `o2m.worldmodel` · `scripts/07–09` | `o2m.splat` / `o2m.render` · `scripts/01–06` |

The splat route still works and is documented under **Reconstruction (legacy
splat)**, but it streaks on this low-texture scene away from the captured path, so
the point-cloud + inpainting route is the one to use for synthetic trajectories.

## Pipeline at a glance (current)

```
recorded episode ──▶ perturb trajectory ──▶ IK feasibility (success/failure)
                                              │
                        ┌─────────────────────┴─────────────────────┐
                   wrist view                                 third-person view
        (mono-depth point cloud of the real          (URDF robot @ IK joints
         frame → reproject → inpaint holes,            composited over the
         or GenWarp diffusion)                         inpainted ZED clean plate)
                        └─────────────────────┬─────────────────────┘
                                              ▼
                    synthetic episode: actions.csv + realsense_color/ + zed_color/
```

## Wrist-view synthesis strategies

The wrist view is the one where fidelity matters (fine gripper↔object interaction).
Everything below shares the **same** mono-depth point cloud and per-frame shift;
they differ only in how the disocclusions are filled:

| Strategy | Config | Cost | Character |
|---|---|---|---|
| depth-warp + `none` | `fill_method: none` | 0 ms | raw holes (debug) |
| depth-warp + `nearest` | `fill_method: nearest` | ~8 ms | fast; streaks |
| depth-warp + `bilinear` | `fill_method: bilinear` | ~2.5 s | smooth but blurry/slow |
| depth-warp + `edge_aware` | `fill_method: edge_aware` | ~40 ms | Navier–Stokes, respects edges |
| depth-warp + `inpaint` *(default)* | `fill_method: inpaint` | ~20 ms | fast-marching (TELEA); good default |
| **GenWarp** | `wrist_renderer: genwarp` | ~1 s* | diffusion NVS; **cleanest** disocclusions |

\* Measured on an RTX 5070 Ti (fp16): ~1.1 s/frame, so a full 369-frame dual-view
episode is ~9 min. Much slower on weaker GPUs — the old docs quoted ~14–18 s/frame.

Compare them all on one frame with `scripts/09_fill_methods_demo.py`; see
**[Wrist synthesis & GenWarp](genwarp.md)**.

## Start here

- **[Synthetic-data pipeline](synthetic_data.md)** — the end-to-end user guide: run
  it, every knob, and how to sweep many variations.
- **[World model](worldmodel.md)** — what it does and why (the design).
- **[Wrist synthesis & GenWarp](genwarp.md)** — the fill-method / novel-view spectrum.
- **[Representations & extensibility](representations.md)** — point clouds, depth,
  splats, and how to swap the geometry backend.
- **[Usage](usage.md)** — environment + smoke tests for both pipelines.
