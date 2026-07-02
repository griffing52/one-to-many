#!/usr/bin/env python3
"""Stage 08 — build an original-vs-synthetic side-by-side comparison video.

Given a synthetic episode produced by ``scripts/07_synthesize_episode.py`` and the
recorded source episode, writes a 2x2 montage video:

    [ WRIST original | WRIST synthetic ]
    [ ZED   original | ZED   synthetic ]

Example
-------
    PYTHONPATH=src python scripts/08_compare_video.py \
        --synthetic outputs/episode_000/synthetic/left_up_4_2cm \
        --raw /home/.../pick_bag_joe/episode_000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from o2m.render.video import save_mp4  # noqa: E402


def _load(p: Path, size) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB").resize(size))


def _label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 12, 20], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", required=True, help="Synthetic episode dir.")
    ap.add_argument("--raw", required=True, help="Recorded source episode dir.")
    ap.add_argument("--out", default=None, help="Output mp4 (default: <synthetic>/compare.mp4).")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    syn, raw = Path(args.synthetic), Path(args.raw)
    out = Path(args.out) if args.out else syn / "compare.mp4"
    meta = {}
    mp = syn / "perturbation.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    tag = f"{meta.get('perturbation', {}).get('name', syn.name)} [{meta.get('label', '?')}]"

    n = len(sorted((syn / "zed_color").glob("*.png")))
    frames = []
    for i in range(n):
        ow = _label(_load(raw / "realsense_color" / f"{i:06d}.png", (640, 480)), "WRIST original")
        sw = _label(_load(syn / "realsense_color" / f"{i:06d}.png", (640, 480)), f"WRIST synth {tag}")
        oz = _label(_load(raw / "zed_color" / f"{i:06d}.png", (640, 360)), "ZED original")
        sz = _label(_load(syn / "zed_color" / f"{i:06d}.png", (640, 360)), f"ZED synth {tag}")
        top = np.concatenate([ow, sw], axis=1)
        bot = np.concatenate([oz, sz], axis=1)
        frames.append(np.concatenate([top, bot], axis=0))

    save_mp4(frames, out, fps=args.fps)
    print(f"wrote {out}  ({n} frames, {tag})")


if __name__ == "__main__":
    main()
