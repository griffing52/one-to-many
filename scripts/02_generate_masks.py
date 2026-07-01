#!/usr/bin/env python3
"""Stage 02 — mask dynamic foreground (arm / object) for COLMAP & splat.

Writes COLMAP-convention masks (black = ignore) to outputs/<ep>/masks/ named
``<image>.png`` (e.g. 000000.png.png).
"""
from __future__ import annotations

import cv2

from _common import base_parser, load, log

from o2m.masking import build_masker, io as mask_io


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--masker", default="roi",
                   choices=["roi", "color", "sam", "grounded_sam"])
    args = p.parse_args()
    cfg, paths = load(args)

    image_paths = sorted(paths.frames.glob("*.png"))
    if not image_paths:
        raise SystemExit("No frames found; run 01_extract_frames.py first.")

    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in image_paths]
    masker = build_masker(args.masker)
    foreground = masker.mask_sequence(images)

    names = [p.name for p in image_paths]
    mask_io.write_colmap_masks(foreground, names, paths.masks)
    frac = sum(float(m.mean()) for m in foreground) / len(foreground)
    log.info("Wrote %d masks -> %s (mean foreground %.1f%%)",
             len(foreground), paths.masks, 100 * frac)


if __name__ == "__main__":
    main()
