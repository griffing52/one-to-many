"""Temporally consistent video depth via Video-Depth-Anything (CVPR 2025).

Same DA-v2 backbone as :class:`.mono.MonoDepthEstimator` plus a temporal
attention head: one call over the whole frame sequence returns disparity maps
that share a consistent relative scale/shift across frames (windowed at 32
frames internally, windows aligned by scale/shift on overlap keyframes). Use
this wherever depth drives a per-frame VIDEO product (the wrist depth-warp);
keep the mono estimator for single-frame uses (ZED scene cloud, dense-init).

Setup (mirrors the GenWarp integration, see ``docs/genwarp.md``):
  - repo cloned at ``a2l/video-depth-anything`` (added to ``sys.path`` lazily),
  - checkpoint ``checkpoints/video_depth_anything_<encoder>.pth`` (get_weights.sh).

Runs without xformers (guarded imports fall back to plain attention).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Repo location (edit here if the repo moves).
_VDA_REPO = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/video-depth-anything")

# Head widths per DINOv2 encoder (from the repo's run.py).
_ENCODER_CFG = {
    "vits": dict(features=64, out_channels=[48, 96, 192, 384]),
    "vitb": dict(features=128, out_channels=[96, 192, 384, 768]),
    "vitl": dict(features=256, out_channels=[256, 512, 1024, 1024]),
}


class VideoDepthEstimator:
    """Wraps VideoDepthAnything.infer_video_depth over an in-memory frame list.

    Returns affine-invariant relative DISPARITY per frame (same convention as
    the HF Depth-Anything pipeline) — convert with
    :func:`o2m.worldmodel.wrist_warp.disparities_to_depths` (one global
    normalisation; per-frame normalisation would reintroduce flicker).
    """

    def __init__(self, encoder: str = "vits", checkpoint: Optional[str] = None,
                 device: Optional[str] = None, repo: Optional[str] = None,
                 input_size: int = 518, fp32: bool = False):
        if encoder not in _ENCODER_CFG:
            raise ValueError(f"encoder must be one of {sorted(_ENCODER_CFG)}")
        self.repo = Path(repo) if repo else _VDA_REPO
        self.encoder = encoder
        self.checkpoint = checkpoint or str(
            self.repo / "checkpoints" / f"video_depth_anything_{encoder}.pth")
        self.input_size = input_size
        self.fp32 = fp32
        self._device = device
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        # The repo imports its own top-level `utils`, which other vendored repos
        # (MVGenMaster's utils.py) can shadow — keep this repo FIRST on sys.path.
        # (Its utils/ also needs an __init__.py: a namespace package loses to any
        # regular utils.py module elsewhere on sys.path regardless of order.)
        p = str(self.repo)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
        import torch
        from video_depth_anything.video_depth import VideoDepthAnything
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VideoDepthAnything(encoder=self.encoder, **_ENCODER_CFG[self.encoder])
        state = torch.load(self.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        self._model = model.to(self._device).eval()

    def estimate_sequence(self, frames: Sequence, fps: float = 30.0) -> np.ndarray:
        """frames: sequence of HxWx3 RGB uint8 arrays (or PIL Images), all the
        same size -> (N, H, W) float32 relative disparity, temporally consistent."""
        self._lazy()
        stack = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
        disp, _ = self._model.infer_video_depth(
            stack, fps, input_size=self.input_size,
            device=self._device, fp32=self.fp32)
        return disp.astype(np.float32)

    def estimate_paths(self, paths: List[Path], fps: float = 30.0) -> np.ndarray:
        from PIL import Image
        return self.estimate_sequence(
            [np.asarray(Image.open(p).convert("RGB")) for p in paths], fps=fps)
