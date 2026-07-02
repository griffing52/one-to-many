# 05 — Rendering

!!! note "Legacy splat renderer"
    This renders the **trained splat**. For synthetic trajectories in the input
    dataset format (point-cloud wrist view + inpainted third-person plate), use
    `scripts/07_synthesize_episode.py` instead — see
    [Generate synthetic episodes](../synthetic_data.md).

`scripts/05_render_trajectory.py` → `o2m.render.RenderPipeline`

**In**: trained splat, `actions.csv`, `align/sim3.json`, `configs/robot.yaml`.
**Out**: `outputs/<ep>/renders/<mode>/<edit>/frame_*.png` + `render.mp4`.

```bash
python scripts/05_render_trajectory.py --episode episode_000 \
    --mode robot_overlay --edit shift_xy
```

| Mode | Behaviour |
|------|-----------|
| `static` | Orbit a free camera through the splat (`Camera`, `orbit_cameras`). |
| `robot_overlay` | Splat from a fixed viewpoint + composited URDF arm. See [robot overlay](06_robot_overlay.md). |
| `dynamic` | Overlay + re-posed object (stretch). |

`--edit` selects a preset from `o2m.trajectory.presets`. `replay_original` uses
the measured joints directly; other edits solve IK per pose.
