"""Monocular-depth priors for densely initialising the splat.

splatfacto (this nerfstudio build) has no depth-supervision loss, so we use
monocular depth a different way: align it to the COLMAP sparse points per frame,
unproject to a dense, geometry-correct seed point cloud, and let splatfacto
initialise its Gaussians from that. This fixes geometry on low-texture surfaces
(blank walls) that pure photometric SfM/splatting leaves under-constrained.
"""
from .mono import MonoDepthEstimator
from .video import VideoDepthEstimator
from .dense_init import build_dense_seed_cloud

__all__ = ["MonoDepthEstimator", "VideoDepthEstimator", "build_dense_seed_cloud"]
