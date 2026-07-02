"""Pure-PyTorch drop-in for ``pesser/splatting`` (softmax forward splatting).

GenWarp's ``genwarp/ops.py`` does ``from splatting import splatting_function``.
The upstream package is a CUDA extension that must be compiled from a git URL; to
keep this self-contained (and to work with newer CUDA than upstream pins), we
provide an equivalent pure-PyTorch implementation and put this directory on
``sys.path`` before importing GenWarp (see ``genwarp_warp.py``).

Softmax splatting (Niklaus & Liu, CVPR 2020): each source pixel is forward-warped
by ``flow`` and accumulated with bilinear weights times ``exp(metric)``; the
output is the weight-normalised sum. Matches the call in ``forward_warper``:
``splatting_function('softmax', image, flow, importance, eps=1e-6)``.
"""
from __future__ import annotations

import torch


def _splat(t: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Forward bilinear scatter of ``t`` (B,C,H,W) by ``flow`` (B,2,H,W dx,dy)."""
    B, C, H, W = t.shape
    dev = t.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                            indexing="ij")
    X = xx[None].float() + flow[:, 0].float()          # B,H,W target x
    Y = yy[None].float() + flow[:, 1].float()          # B,H,W target y
    x0 = torch.floor(X); y0 = torch.floor(Y)
    x1 = x0 + 1; y1 = y0 + 1
    wx1 = X - x0; wx0 = 1.0 - wx1
    wy1 = Y - y0; wy0 = 1.0 - wy1

    out = torch.zeros(B, C, H * W, device=dev, dtype=torch.float32)
    tf = t.float().reshape(B, C, H * W)
    for xc, yc, w in ((x0, y0, wx0 * wy0), (x1, y0, wx1 * wy0),
                      (x0, y1, wx0 * wy1), (x1, y1, wx1 * wy1)):
        valid = (xc >= 0) & (xc <= W - 1) & (yc >= 0) & (yc <= H - 1)
        xi = xc.long().clamp(0, W - 1)
        yi = yc.long().clamp(0, H - 1)
        idx = (yi * W + xi).reshape(B, 1, H * W).expand(B, C, H * W)
        wv = (w * valid).to(torch.float32).reshape(B, 1, H * W)
        out.scatter_add_(2, idx, tf * wv)
    return out.reshape(B, C, H, W)


def splatting_function(mode: str, tensor: torch.Tensor, flow: torch.Tensor,
                       importance_metric: torch.Tensor = None,
                       eps: float = 1e-6) -> torch.Tensor:
    """See module docstring. Supports the modes upstream exposes."""
    if mode == "softmax":
        if importance_metric is None:
            raise ValueError("softmax splatting needs an importance metric")
        w = torch.exp(importance_metric.float())
        num = _splat(tensor * w, flow)
        den = _splat(w.expand_as(tensor[:, :1]) if w.shape[1] == 1 else w, flow)
        out = num / (den + eps)
        return out.to(tensor.dtype)
    if mode in ("summation", "sum"):
        return _splat(tensor, flow).to(tensor.dtype)
    if mode in ("average", "avg"):
        ones = torch.ones_like(tensor[:, :1])
        num = _splat(tensor, flow)
        den = _splat(ones, flow)
        return (num / (den + eps)).to(tensor.dtype)
    if mode in ("linear",):
        if importance_metric is None:
            raise ValueError("linear splatting needs an importance metric")
        num = _splat(tensor * importance_metric.float(), flow)
        den = _splat(importance_metric.float(), flow)
        return (num / (den + eps)).to(tensor.dtype)
    raise ValueError(f"unknown splatting mode {mode!r}")
