"""GenWarp-based wrist-view synthesis (generative novel-view alternative).

An alternative to the depth-warp wrist renderer (:mod:`.wrist_warp`) that uses
Sony's **GenWarp** (NeurIPS 2024) to *generate* the shifted view with a diffusion
model instead of only reprojecting real pixels. GenWarp warps the source by depth
and then a semantic-preserving diffusion model fills the disocclusions, so the
shifted image stays clean where depth-warp sprays holes on thin foreground.

Setup (see ``docs/genwarp.md``):
  - repo cloned at ``a2l/genwarp`` (this wrapper adds it + a pure-torch ``splatting``
    shim to ``sys.path`` at call time),
  - checkpoints under ``a2l/genwarp/checkpoints`` (``multi1`` + ``image_encoder`` +
    ``sd-vae-ft-mse``), fetched by ``genwarp/scripts/download_models.sh``.

Geometry: GenWarp's world is z-up / y-right / x-back with the source camera at the
origin looking ``-x``. Our optical offset ``dcam`` (x-right, y-down, z-forward) maps
to a world eye offset ``[-dz, dx, -dy]``; the target camera is that translation with
the *same* look direction (pure parallax, matching the rigid wrist shift).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Repo + shim locations (edit here if the repo moves).
_GENWARP_REPO = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/genwarp")
_SHIM_DIR = Path(__file__).resolve().parent / "_genwarp_ext"


def _ensure_paths(repo: Path = _GENWARP_REPO) -> None:
    for p in (str(_SHIM_DIR), str(repo)):   # shim first so `import splatting` hits it
        if p not in sys.path:
            sys.path.insert(0, p)


class GenWarpWrapper:
    def __init__(self, checkpoint_dir: Optional[str] = None,
                 checkpoint_name: str = "multi1", half: bool = True,
                 num_inference_steps: int = 20, guidance_scale: float = 3.5,
                 repo: Optional[str] = None):
        self.repo = Path(repo) if repo else _GENWARP_REPO
        self.ckpt_dir = checkpoint_dir or str(self.repo / "checkpoints")
        self.checkpoint_name = checkpoint_name
        self.half = half
        self.steps = num_inference_steps
        self.guidance = guidance_scale
        self._model = None
        self._ops = None

    def _lazy(self):
        if self._model is not None:
            return
        _ensure_paths(self.repo)
        import torch
        from genwarp import GenWarp
        from genwarp import ops as gw_ops
        cfg = dict(pretrained_model_path=self.ckpt_dir,
                   checkpoint_name=self.checkpoint_name,
                   half_precision_weights=self.half,
                   num_inference_steps=self.steps,
                   guidance_scale=self.guidance)
        self._model = GenWarp(cfg=cfg)
        self._ops = gw_ops
        self._torch = torch

    def warp(self, real_rgb: np.ndarray, depth: np.ndarray, dcam: np.ndarray,
             fy: float, res: int = 512, return_aux: bool = False,
             mode: str = "pad", depth_scale: float = 1.0):
        """Generate the shifted wrist view.

        GenWarp shares the SAME depth and camera offset as the depth-warp path — it
        only differs in that a diffusion model fills the disocclusions. The knobs:

        Args:
            real_rgb: HxWx3 uint8 source frame.
            depth: HxW positive depth (same :func:`.wrist_warp.disparity_to_depth`
                the depth-warp uses).
            dcam: (3,) optical-frame camera translation (m).
            fy: source vertical focal length (px). fovy is derived per-mode so the
                parallax matches the depth-warp exactly.
            mode: ``pad`` (letterbox to a square — keeps the FULL frame, recommended),
                  ``crop`` (centre square, drops the sides), or ``squash`` (stretch).
            depth_scale: multiply depth before warping. Parallax ~ dcam/depth, so
                <1 exaggerates the shift, >1 shrinks it. The main shift-tuning dial
                when the relative depth's scale is off.
        Returns:
            HxWx3 uint8 synthesized frame (dict with 'warped' too if return_aux).
        """
        self._lazy()
        torch = self._torch
        ops = self._ops
        H0, W0 = real_rgb.shape[:2]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dt = torch.float16 if (self.half and dev == "cuda") else torch.float32
        dep_in = np.ascontiguousarray(depth) * float(depth_scale)

        # Make a square canvas so GenWarp's 512x512 model sees undistorted geometry.
        # ``sq_half`` (original px from the optical centre to the square's edge) sets
        # fovy, so the induced parallax equals the depth-warp's for the same dcam.
        if mode == "crop":
            side = min(H0, W0)
            y0, x0 = (H0 - side) // 2, (W0 - side) // 2
            rgb_sq = real_rgb[y0:y0 + side, x0:x0 + side]
            dep_sq = dep_in[y0:y0 + side, x0:x0 + side]
            sq_half = side / 2.0
        elif mode == "pad":
            side = max(H0, W0)
            y0, x0 = (side - H0) // 2, (side - W0) // 2
            rgb_sq = np.zeros((side, side, 3), np.uint8)
            rgb_sq[y0:y0 + H0, x0:x0 + W0] = real_rgb
            dep_sq = np.full((side, side), float(np.median(dep_in)), np.float32)
            dep_sq[y0:y0 + H0, x0:x0 + W0] = dep_in
            sq_half = side / 2.0
        else:  # squash
            side = None
            rgb_sq, dep_sq = real_rgb, dep_in
            sq_half = H0 / 2.0

        img = torch.from_numpy(rgb_sq.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        img = torch.nn.functional.interpolate(img, (res, res), mode="bilinear",
                                              align_corners=False).to(dev, dt)
        dep = torch.from_numpy(np.ascontiguousarray(dep_sq)).float()[None, None]
        dep = torch.nn.functional.interpolate(dep, (res, res), mode="nearest").to(dev, dt)

        fovy = 2.0 * float(np.arctan(sq_half / float(fy)))
        proj = ops.get_projection_matrix(
            fovy=torch.ones(1) * fovy, aspect_wh=1.0, near=0.01, far=100.0
        ).to(dev, dt)

        z_up = torch.tensor([[0.0, 0.0, 1.0]])
        src_view = ops.camera_lookat(torch.tensor([[0.0, 0.0, 0.0]]),
                                     torch.tensor([[-1.0, 0.0, 0.0]]), z_up)
        d = np.asarray(dcam, float)
        eye = torch.tensor([[-d[2], d[0], -d[1]]], dtype=torch.float32)   # optical->world
        look = eye + torch.tensor([[-1.0, 0.0, 0.0]])                     # same direction
        tar_view = ops.camera_lookat(eye, look, z_up)
        rel = (tar_view @ torch.linalg.inv(src_view.float())).to(dev, dt)

        with torch.no_grad():
            renders = self._model(src_image=img, src_depth=dep, rel_view_mtx=rel,
                                  src_proj_mtx=proj, tar_proj_mtx=proj)

        def _to_full(t):
            """Diffusion 512x512 tensor -> uint8 frame at the source resolution."""
            sz = (side, side) if side else (H0, W0)
            arr = torch.nn.functional.interpolate(
                t.float().clamp(0, 1), sz, mode="bilinear",
                align_corners=False)[0].permute(1, 2, 0).cpu().numpy()
            arr = (arr * 255).astype(np.uint8)
            if side is None:
                return arr
            return arr[y0:y0 + H0, x0:x0 + W0]      # crop the square back to HxW

        out = _to_full(renders["synthesized"])
        if not return_aux:
            return out
        return {"synthesized": out, "warped": _to_full(renders["warped"])}
