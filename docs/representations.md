# Representations & extensibility

The goal is an **abstract, swappable pipeline**: capture → poses/depth → *some 3D
representation* → render. The representation is **not** hard-wired to Gaussian
Splatting. The current synthetic-data pipeline uses **per-frame monocular-depth
point clouds** (lift the real frame, reproject, inpaint) precisely because 3DGS
streaks on this low-texture scene. A trained splat is still available for
free-viewpoint flythroughs.

## Representations available today

| Representation | How | Robust in low texture? | Used by |
|---|---|---|---|
| **Mono-depth point cloud** *(current default)* | `disparity_to_depth(DA-v2)` per frame → unproject → reproject (`o2m.worldmodel.wrist_warp`) | detail preserved from the real frame; approximate metric scale | wrist-view synthesis (depth-warp / GenWarp) |
| **Inpainted clean plate** | ZED f0 with the robot painted out (SAM mask + inpaint) | exact (it's the real background) | third-person view (URDF arm over the plate) |
| **Gaussian splat** (checkpoint) | `04_train_splat` | RGB sharp from *captured* views; novel views streak | legacy free-viewpoint / `robot_overlay` |
| **Gaussian point cloud** (`pointcloud.ply`) | `export_gaussian_pointcloud` (from the splat, no pymeshlab) | geometry only as good as the splat | quick geometry handle, viewers, collision |
| **Depth maps** | `SplatModel.render` → `colorize_depth`, or DA-v2 directly | noisy on textureless surfaces | occlusion, unprojection |
| **Dense unprojected cloud** | `unproject_depth_to_points` | inherits depth noise | per-view dense points |
| **COLMAP sparse cloud** (`sparse_pc.ply`) | stage 03 | only textured points | SfM sanity, splat seed |

All are written under `outputs/<ep>/` and need no GUI.

## Why point clouds + inpainting instead of a splat

For the actual use case — **small-to-moderate perturbations near the recorded
path** — lifting the *real* frame to a point cloud and reprojecting keeps the full
recorded detail (sharp shelf, legs, handles, bag), while a splat trained on this
scene:

- is under-constrained where there is no texture (photometric 3DGS can put
  Gaussians at *any* depth on a flat wall → swirly depth on the white walls), and
- streaks badly at any pose away from the sparse registered keyframes
  (low-texture + fast wrist motion registers few frames).

So the splat looks fine only *at* captured views; the point-cloud reprojection is
exact for the shift and degrades gracefully. The trade-off: a single-frame point
cloud has **disocclusion holes** where the shift reveals hidden geometry — that is
what the **inpainting / hole-fill** stage exists to fill (see below and
[Wrist synthesis & GenWarp](genwarp.md)).

### Getting more reliable geometry (future captures)

1. **Record RGB-D.** The RealSense D435i *has* a depth sensor; this dataset didn't
   save it. Real depth removes the mono-depth scale ambiguity and the hole-filling
   guesswork — the single highest-impact change.
2. **Metric mono-depth.** Swap DA-v2-small for DA-v2-**metric** for a physically
   scaled shift (then `genwarp.depth_scale: 1.0`).
3. **Multi-frame fusion.** Fill wrist disocclusion holes from *neighbouring* wrist
   frames instead of inpainting — the real fix for large offsets.

## The inpainting / hole-fill layer

"Inpainting" shows up in three places in the current pipeline:

1. **Wrist disocclusion fill** — after reprojecting the point cloud, the newly
   revealed pixels are filled. Strategies (`warp.fill_method`): `none | nearest |
   bilinear | edge_aware(Navier–Stokes) | inpaint(TELEA, default)`, or hand the
   whole job to **GenWarp** (`wrist_renderer: genwarp`), a diffusion model that
   *generates* coherent content instead of interpolating.
2. **Clean plate** — the third-person background is the ZED frame with the robot
   **segmented (SAM) and inpainted out**, so the URDF arm can be composited over a
   robot-free scene.
3. **Vacated-object spot** *(stretch)* — when an object is re-posed, inpaint the
   spot it left (see [dynamic objects](stretch_dynamic.md)).

## Swapping the representation

The pipeline is layered so a new representation only replaces the middle:

```
data (Episode) ─▶ per-frame mono-depth ─▶ point cloud ─▶ reproject ─▶ inpaint  (wrist)
data (Episode) ─▶ ZED→base calib ───────▶ clean plate ─▶ URDF composite        (3rd-person)

                (legacy)  masks ─▶ COLMAP poses ─▶ splatfacto ─▶ splat renders
```

To add a wrist backend, expose `warp(real_rgb, depth, dcam) -> rgb` like
`WristWarper` / `GenWarpWrapper`; to add a scene backend for the third-person view,
expose `render(camera) -> (rgb, depth, alpha)` like `SplatModel`. Poses, masks,
alignment (`sim3` / ZED extrinsic), perturbation, and IK feasibility are all
representation-agnostic and stay unchanged.
