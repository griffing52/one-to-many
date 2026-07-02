# 04 — Splat training

!!! note "Legacy / free-viewpoint route"
    Splat training is **not** used by the current synthetic-data pipeline, which
    reprojects mono-depth point clouds and inpaints instead (see
    [Synthetic-data pipeline](../synthetic_data.md) and
    [Representations](../representations.md)). Train a splat only for free-viewpoint
    flythroughs of the static scene; it streaks away from the captured path.

`scripts/04_train_splat.py` → `o2m.splat`

**In**: `outputs/<ep>/nerfstudio/` (transforms.json + images + masks + seed ply).
**Out**: `outputs/<ep>/splat/config.yml` (+ checkpoint) and `splat/splat.ply`.

Wraps `ns-train splatfacto` then `ns-export gaussian-splat`. Smoke test with a
few thousand iterations; full quality at 30 000.

```bash
python scripts/04_train_splat.py --episode episode_000 --max-iters 2000
```

!!! warning
    Auto-orient / auto-scale are **disabled** so the splat stays in the COLMAP
    frame. This is what keeps the [alignment](../alignment.md) valid — do not
    re-enable them.
