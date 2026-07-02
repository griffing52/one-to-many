# 01 — Frame extraction

`scripts/01_extract_frames.py` → `o2m.data.Episode`

**In**: raw `episode_NNN/` (wrist + zed PNGs, `actions.csv`).
**Out**: `outputs/<ep>/frames/000000.png …` (COLMAP image set) and
`frame_index.json` recording the subsampling stride and the selected raw frame
indices.

```bash
python scripts/01_extract_frames.py --episode episode_000 --frame-stride 4
```

The wrist (moving) camera provides the multi-view parallax needed for
reconstruction. `frame_index.json` is essential later: it maps each extracted
frame back to its CSV row so measured joints line up with COLMAP poses during
[alignment](../alignment.md).
