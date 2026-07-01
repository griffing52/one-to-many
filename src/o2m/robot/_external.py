"""Locate and import the existing kinematics / scene utilities living at the
``bot2bot`` repo root (outside this package), so we reuse them rather than
re-implementing FK/IK and URDF->MuJoCo loading.

Reused modules:
- ``kinematic_translator``      (initialize_models, get_ur5e_ee_pose, solve_piper_ik)
- ``cross_embodiment.mujoco_scene`` (load_urdf_with_assets, build_joint_index)
- ``cross_embodiment.rendering``    (save_frames_png, save_animation)
"""
from __future__ import annotations

import sys
from pathlib import Path

# .../bot2bot/bot2bot/a2l/one-to-many/src/o2m/robot/_external.py
# repo root containing kinematic_translator.py and cross_embodiment/ is bot2bot/
_BOT2BOT_ROOT = Path(__file__).resolve().parents[6]


def ensure_on_path() -> Path:
    root = str(_BOT2BOT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _BOT2BOT_ROOT
