"""Dynamic-foreground masking for COLMAP / splat training.

A masker returns a boolean array where ``True`` marks dynamic foreground
(robot arm, gripper, manipulated object) that must be EXCLUDED from SfM and
splat optimisation. ``io.write_colmap_masks`` writes the COLMAP convention
(black = ignore) to disk.
"""
from .base import Masker
from .color_motion import ColorMotionMasker
from .roi_masker import ROIMasker
from . import io

__all__ = ["Masker", "ColorMotionMasker", "ROIMasker", "io", "build_masker"]


def build_masker(name: str, **kwargs) -> Masker:
    name = name.lower()
    if name in ("roi", "gripper"):
        return ROIMasker(**kwargs)
    if name in ("color", "motion", "color_motion"):
        return ColorMotionMasker(**kwargs)
    if name in ("sam", "grounded_sam", "grounded-sam"):
        from .sam_masker import GroundedSAMMasker
        return GroundedSAMMasker(**kwargs)
    raise ValueError(f"Unknown masker: {name!r}")
