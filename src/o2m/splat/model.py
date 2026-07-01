"""Render a trained splat from an arbitrary :class:`Camera`.

Loads a nerfstudio checkpoint (preferred, full gsplat quality) via the saved
config.yml, or an exported ``.ply``. ``render`` returns (rgb, depth, alpha) in
the camera frame so the overlay pipeline can do depth-aware compositing.

The nerfstudio internals are imported lazily so the core package does not depend
on the (large, CUDA-specific) splat stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .camera import Camera


class SplatModel:
    def __init__(self, pipeline=None, ply_path: Optional[Path] = None):
        self._pipeline = pipeline
        self._ply_path = ply_path

    # --- loading -----------------------------------------------------------
    @classmethod
    def from_config(cls, config_yml: Path, device: str = "cuda") -> "SplatModel":
        from nerfstudio.utils.eval_utils import eval_setup  # lazy
        import torch

        # Ensure in-process gsplat JIT compilation (at render time) can find
        # ninja/nvcc and stays RAM-bounded.
        from .train import ensure_build_env_in_process
        ensure_build_env_in_process()

        # nerfstudio (older) calls torch.load without weights_only; torch>=2.6
        # defaults weights_only=True and rejects the numpy globals in the
        # checkpoint. We trust our own checkpoint, so force weights_only=False
        # for the duration of loading.
        _orig_load = torch.load

        def _trusting_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _trusting_load
        try:
            _, pipeline, _, _ = eval_setup(Path(config_yml))
        finally:
            torch.load = _orig_load
        pipeline.model.to(torch.device(device))
        pipeline.model.eval()
        return cls(pipeline=pipeline)

    @classmethod
    def from_ply(cls, ply_path: Path) -> "SplatModel":
        return cls(ply_path=Path(ply_path))

    # --- rendering ---------------------------------------------------------
    def render(self, camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render the splat from ``camera``; returns (rgb uint8, depth, alpha)."""
        if self._pipeline is None:
            raise RuntimeError(
                "SplatModel.render needs a loaded nerfstudio pipeline. Either "
                "load via from_config(), or render the .ply with an external "
                "gsplat rasteriser."
            )
        import torch
        from nerfstudio.cameras.cameras import Cameras, CameraType

        c2w = self._to_nerfstudio_c2w(camera)
        cams = Cameras(
            camera_to_worlds=torch.tensor(c2w[None, :3, :], dtype=torch.float32),
            fx=torch.tensor([[camera.fx]]), fy=torch.tensor([[camera.fy]]),
            cx=torch.tensor([[camera.cx]]), cy=torch.tensor([[camera.cy]]),
            width=torch.tensor([[camera.width]]), height=torch.tensor([[camera.height]]),
            camera_type=CameraType.PERSPECTIVE,
        ).to(self._pipeline.device)

        with torch.no_grad():
            outputs = self._pipeline.model.get_outputs_for_camera(cams)

        rgb = (outputs["rgb"].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = outputs.get("depth")
        alpha = outputs.get("accumulation")
        depth = depth.cpu().numpy()[..., 0] if depth is not None else np.zeros(rgb.shape[:2])
        alpha = alpha.cpu().numpy()[..., 0] if alpha is not None else np.ones(rgb.shape[:2])
        return rgb, depth, alpha

    @staticmethod
    def _to_nerfstudio_c2w(camera: Camera) -> np.ndarray:
        """Nerfstudio Cameras expect OpenGL camera axes; convert from OpenCV."""
        from ..utils import geometry as geom
        return geom.opencv_c2w_to_opengl(camera.c2w)
