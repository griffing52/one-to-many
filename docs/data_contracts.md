# Data contracts

Authoritative spec for the two interchange formats. Implemented in
`o2m/colmap/` and `o2m/masking/io.py`.

## COLMAP input

- **Images**: subsampled wrist frames in `outputs/<ep>/frames/`, sequentially
  named `000000.png …`. All share one physical camera → `--single_camera 1`.
- **Camera model**: `OPENCV` (`fx, fy, cx, cy, k1, k2, p1, p2`), seeded with the
  priors in `configs/camera.yaml`.
- **Masks**: COLMAP ignores pixels where the mask is **black (0)** and uses
  **white (255)**. Mask filename = `<image_filename>.png`, i.e. for
  `images/000123.png` the mask is `masks/000123.png.png`. Our maskers return
  `True` for *foreground to exclude*, so `io.write_colmap_masks` writes the
  inverse (foreground → 0).
- **Matching**: `sequential_matcher` (frames are an ordered video).

Consumed outputs: `cameras.bin` (intrinsics), `images.bin` (per-image world→cam
`qvec`/`tvec`, OpenCV), `points3D.bin` (sparse seed cloud).

## Nerfstudio `transforms.json`

Single shared camera → intrinsics at the top level:

```json
{
  "camera_model": "OPENCV",
  "fl_x": 600.0, "fl_y": 600.0, "cx": 320.0, "cy": 240.0,
  "w": 640, "h": 480,
  "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
  "ply_file_path": "sparse_pc.ply",
  "frames": [
    {
      "file_path": "images/000000.png",
      "mask_path": "masks/000000.png.png",
      "transform_matrix": [[r,r,r,t],[r,r,r,t],[r,r,r,t],[0,0,0,1]]
    }
  ]
}
```

### Convention conversion (must be exact)

1. COLMAP gives world→cam `R_wc, t_wc` (OpenCV).
2. cam→world: `R_cw = R_wc.T`, `C = -R_wc.T @ t_wc`.
3. Nerfstudio wants **OpenGL** camera axes → flip local y,z:
   `M_gl = M_cv @ diag(1,-1,-1,1)`. Store `M_gl` as `transform_matrix`.

The same masks are carried as `mask_path` so the trained splat excludes arm
pixels too.

!!! warning "Keep the frame fixed"
    Train with nerfstudio auto-orient / auto-scale **disabled**
    (`orientation_method none`, `center_method none`, `auto_scale_poses false`).
    The splat world frame must equal the COLMAP frame, otherwise the
    [sim3 alignment](alignment.md) is invalid.
