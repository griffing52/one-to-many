# 02 — Masking

`scripts/02_generate_masks.py` → `o2m.masking`

**In**: `outputs/<ep>/frames/`.
**Out**: `outputs/<ep>/masks/<image>.png` in COLMAP convention (black = ignore).

The arm and manipulated object are *dynamic foreground*: if COLMAP matches
features on them, the recovered poses and scale are corrupted. A masker returns
`True` for foreground; `io.write_colmap_masks` writes the inverse.

| Masker | Notes |
|--------|-------|
| `color` *(default)* | Model-free temporal-median motion mask. Good bootstrap. |
| `grounded_sam` | Text-prompted segmentation (`"robot arm"`, `"gripper"`, `"bag"`). Higher quality; needs the `mask` extra. |

```bash
python scripts/02_generate_masks.py --episode episode_000 --masker color
```

The same masks are reused as `mask_path` in `transforms.json` so the trained
splat also excludes arm pixels.
