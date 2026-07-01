"""Frame/video writers. Thin re-export of the shared cross_embodiment helpers,
with a local fallback so the core package works without that import.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def save_frames_png(frames: List[np.ndarray], out_dir: Path, stem: str = "frame") -> List[Path]:
    import imageio.v2 as imageio
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        p = out_dir / f"{stem}_{i:04d}.png"
        imageio.imwrite(p, frame)
        paths.append(p)
    return paths


def save_mp4(frames: List[np.ndarray], path: Path, fps: int = 30) -> Path:
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)
    return path
