# Alignment (the crux)

Both pipelines must place the metric robot into an image frame. There are two
routes, each solving a different unknown:

- **Splat route (this page):** recover the sim3 **`T_splat←base`** between the
  arbitrary COLMAP/splat world and the robot base — needed for `robot_overlay`.
- **Synthetic-data route:** calibrate the fixed **ZED into the base frame** by
  target-free PnP on the tracked green gripper (~7 px), persisted at
  `outputs/<ep>/align/zed_extrinsic.npz`. No splat, no COLMAP. This is what the
  current [synthetic-data pipeline](synthetic_data.md) uses.

The rest of this page is the splat sim3.

To draw the URDF arm inside the splat we need the similarity transform
**`T_splat←base`** (scale + rotation + translation) mapping robot-base
coordinates into the splat/COLMAP world. Implemented in `o2m.align.world`.

## Why a similarity (sim3)?

COLMAP reconstructs up to an arbitrary global scale and pose. The robot's
EE/joint data is metric, in the base frame. So the two frames differ by a
rotation, a translation, **and a scale** — exactly a sim(3).

## Free correspondences from the wrist camera

The wrist camera is rigidly mounted on the arm. For each reconstructed frame *i*:

- `C_i` = COLMAP wrist-camera centre **in the splat frame**
  (from `transforms.json`);
- `B_i` = FK of the URDF **camera link** at the measured joints **in the base
  frame** (`PiperModel.camera_pose_base`).

With *N* ≈ hundreds of frames, `umeyama_sim3(B, C)` solves for `(s, R, t)` such
that `C ≈ s·R·B + t`. The recovered `s` converts COLMAP units to metres.

```python
sim3, diag = WorldAligner.from_wrist_fk(colmap_centers, fk_centers_base)
# diag = {scale, residual_rms (m), residual_max, n}
```

## Using it when rendering

For `robot_overlay` we render the splat from a viewpoint in the splat frame, and
render the arm (metric, base frame) from the **same** physical viewpoint by
mapping the camera back with `Sim3.inv_apply`:

```
camera_base.c2w = sim3.inv_apply(camera_splat.c2w)
```

## Diagnostics & failure modes

- **High `residual_rms`** (≳ 5 cm): usually bad masks letting COLMAP match the
  moving arm, or the URDF camera link not matching the physical mount. Tighten
  masks (switch `color → sam`), or set `align.cam_offset_*` in `robot.yaml`.
- **Implausible scale**: COLMAP degenerate reconstruction — check the registered
  fraction and reprojection error from stage 03.
- **Fallback**: `WorldAligner.from_known_points` aligns from a few manual
  correspondences (e.g. tabletop corners) if the FK-camera route is unreliable.

## Registering the ZED viewpoint

The stationary ZED extrinsic is unknown. To render `robot_overlay` from the demo
viewpoint, register one ZED frame into the splat frame (PnP against the sparse
model, or add ZED frames as a second camera in COLMAP) and save it to
`outputs/<ep>/align/zed_camera.json`. Until then, stage 05 falls back to a wrist
viewpoint.
