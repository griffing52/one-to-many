# Wrist synthesis — warp, inpaint & GenWarp

The wrist view is synthesised **without a splat**: the real frame is lifted to a
monocular-depth (DA-v2) **point cloud**, a virtual camera is moved by the per-frame
shift, and the cloud is reprojected. Forward warping a single-frame cloud leaves
**disocclusion holes** (newly-visible regions behind foreground). We support a
spectrum of ways to fill them, from cheap interpolation to a diffusion novel-view
model:

| Method | `wrist_renderer` / `fill_method` | Cost | Character |
|---|---|---|---|
| No fill | `fill_method: none` | 0 ms | shows the raw holes (debug) |
| Nearest | `fill_method: nearest` | ~8 ms | fast; streaks along holes |
| Bilinear | `fill_method: bilinear` | ~2.5 s | smooth but blurry & slow (griddata) |
| Edge-aware | `fill_method: edge_aware` | ~40 ms | Navier–Stokes inpaint, respects edges |
| Inpaint (TELEA) | `fill_method: inpaint` *(default)* | ~20 ms | fast-marching inpaint; good default |
| **GenWarp** | `wrist_renderer: genwarp` | ~1 s* | diffusion NVS; **cleanest** disocclusions |

\* ~1.1 s/frame measured on an RTX 5070 Ti (fp16); a full 369-frame dual-view
episode is ~9 min. Older docs quoted ~14–18 s/frame (a much weaker GPU) — it is
GPU-bound, so scale accordingly.

All are geometry-consistent (same depth-warp), so they differ only in how the holes
are filled. Compare them on one frame:

```bash
PYTHONPATH=src MUJOCO_GL=egl python scripts/09_fill_methods_demo.py \
    --frame 40 --offset 0 0.04 0
# -> outputs/episode_000/renders/worldmodel/fill_methods_f40.png
```

## GenWarp (Sony, NeurIPS 2024)

GenWarp generates the shifted view with a semantic-preserving diffusion model: it
depth-warps the source, then a UNet fills disocclusions with plausible, coherent
content instead of smeared pixels. Best for larger shifts where interpolation
sprays. Trade-off: mild generative drift, and it is GPU-bound (~1 s/frame on an
RTX 5070 Ti; much more on weaker cards).

### Does it use the same depth/warp as the depth-warp path?

**Depth: yes — identical.** GenWarp is fed the *same* `disparity_to_depth(DA-v2)`
map and the *same* camera offset `dcam` as the depth-warp renderer. It does **not**
estimate its own depth.

**Warp: it forward-warps internally, and it must.** GenWarp's diffusion UNet is
*trained conditioned on* GenWarp's own forward-warp + correspondence + positional
embeddings (`prepare_conditions` in `genwarp/GenWarp.py`). Feeding it our
`_scatter` warp instead would be out-of-distribution for the model. But that
internal warp runs on *our* depth and *our* camera matrices, so it is the **same
geometry** as the depth-warp — we made the projection match exactly (`fovy` derived
from the same `fy` and the padded square), so GenWarp's shift equals the
depth-warp's shift; GenWarp only differs in that it *generates* the holes instead
of interpolating them.

### Setup (one time)

The repo is cloned at `a2l/genwarp`. Two things are needed:

1. **Checkpoints** (~7 GB for `multi1`) — download into `a2l/genwarp/checkpoints`:
   ```bash
   cd a2l/genwarp && ./scripts/download_models.sh ./checkpoints
   ```
   Gives `multi1/`, `image_encoder/`, `sd-vae-ft-mse/`.
2. **`splatting`** — GenWarp's `ops.py` imports `splatting_function` from the
   `pesser/splatting` CUDA extension. Rather than build it, we ship an equivalent
   **pure-PyTorch** implementation at
   `src/o2m/worldmodel/_genwarp_ext/splatting.py` and put it on `sys.path` before
   importing GenWarp (see `genwarp_warp.py`). No compilation, works with new CUDA.

Deps (`diffusers accelerate omegaconf einops roma`) install into the `o2m` conda
env; GenWarp imports fine under diffusers 0.38 / transformers 5.12.

### How it's wired

`src/o2m/worldmodel/genwarp_warp.py :: GenWarpWrapper` converts our optical-frame
offset `dcam` (x-right, y-down, z-forward) into GenWarp's world (z-up, y-right,
x-back; source camera at origin looking `-x`): eye offset `[-dz, dx, -dy]`, target
looks the same direction (pure parallax). The wrist frame is **centre-cropped to a
square** (GenWarp is 512², so cropping avoids the 4:3→1:1 aspect distortion),
generated, and pasted back; the gripper trapezoid is restored from the original.

Depth is the same `disparity_to_depth` used by the depth-warp path, so the two
renderers produce matched parallax.

### Use it in the pipeline

```bash
# whole synthetic episode with the GenWarp wrist renderer (slow):
PYTHONPATH=src MUJOCO_GL=egl python scripts/07_synthesize_episode.py \
    --wrist-renderer genwarp --name left_4cm_genwarp

# or set worldmodel.wrist_renderer: genwarp in configs/worldmodel.yaml
```

### Tuning the shift (`configs/worldmodel.yaml → worldmodel.genwarp:`)

| Knob | Effect |
|---|---|
| `depth_scale` | **Main shift dial.** Parallax ≈ `dcam / depth`, so `<1` exaggerates the shift, `>1` shrinks it. Use it when the relative (non-metric) depth's scale is off. |
| `mode` | `pad` (letterbox → keeps the **full** frame, matches the depth-warp framing — default), `crop` (centre square, drops the sides), `squash` (stretch, distorts). |
| `num_inference_steps` | More steps → cleaner / less noisy, slower. |
| `guidance_scale` | Higher → sticks to the warp more (less hallucinated drift); lower → smoother. |
| the offset itself | `perturb.base_offset` — the physical shift the whole pipeline applies. |

For a *metric* shift magnitude, swap DA-v2-small for **DA-v2-metric** (see
`examples/genwarp_inference_dav2.ipynb`) and set `depth_scale: 1.0`.

### Where to edit

| What | Where |
|---|---|
| Repo / checkpoint paths | `src/o2m/worldmodel/genwarp_warp.py` (`_GENWARP_REPO`) |
| Shift knobs (mode / depth_scale / steps / guidance) | `configs/worldmodel.yaml → worldmodel.genwarp` |
| splatting shim | `src/o2m/worldmodel/_genwarp_ext/splatting.py` |
| fill method (depth-warp) | `configs/worldmodel.yaml → worldmodel.warp.fill_method` |

### Known caveats
- GPU-bound: ~1.1 s/frame on an RTX 5070 Ti (full 369-frame dual-view episode ~9 min),
  but ~14–18 s/frame on the weak GPU the docs were first written on. `depthwarp` is
  still ~10× faster if you don't need the clean disocclusions.
- Diffusion regenerates each frame independently, so expect mild temporal flicker /
  texture drift across a video (a known NVS trait).
- Depth is relative (DA-v2 small); the shift magnitude is approximate — tune with
  `depth_scale`, or swap in DA-v2 *metric* for a physically-scaled shift.
