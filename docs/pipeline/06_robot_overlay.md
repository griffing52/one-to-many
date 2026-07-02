# 06 — Robot overlay

!!! note "Legacy splat overlay"
    This composites the arm over **splat** renders. The current third-person view
    composites the same URDF arm over an **inpainted ZED clean plate** at
    IK joints (calibrated by PnP, no splat) — see
    [Generate synthetic episodes](../synthetic_data.md) and `o2m.worldmodel.thirdperson`.
    The FK/IK/render machinery below is shared by both.

A kinematically-driven Piper arm composited into splat renders. Modules:
`o2m.robot` (URDF/FK/IK/render), `o2m.align` (sim3), `o2m.render` (composite).

## Steps per frame

1. **Edit** the EE trajectory (`o2m.trajectory`) → new EE poses in the base
   frame.
2. **Joints**: measured joints for `replay_original`, else IK (`q_for_ee`).
3. **Render splat** from the viewpoint (splat frame) → background RGB (+ depth).
4. **Render arm** (`RobotRenderer.render_rgba`) from the *same* viewpoint mapped
   into the base frame via `Sim3.inv_apply` → RGB + alpha (+ depth).
5. **Composite** (`composite_rgba_over`), optionally depth-aware.

## Reused repo assets

- `kinematic_translator` — Pinocchio FK / damped-least-squares IK pattern.
- `cross_embodiment.mujoco_scene.load_urdf_with_assets` — URDF→MuJoCo with
  `package://` mesh resolution.
- `cross_embodiment.rendering` — frame/animation writers.
- `piper_description_v100_camera.urdf` — exposes the wrist-camera link (alignment
  FK) and the gripper finger links.

## Caveats

- The arm is rendered in MuJoCo (no native alpha); alpha comes from the
  segmentation pass.
- Depth-aware occlusion requires reconciling splat depth (scaled) with MuJoCo
  depth (metres); the first version composites by alpha only. See
  `RenderPipeline.render_robot_overlay`.
- The ZED demo viewpoint must be [registered](../alignment.md#registering-the-zed-viewpoint)
  into the splat frame; otherwise a wrist viewpoint is used.
