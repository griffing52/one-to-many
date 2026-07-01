"""Render the Piper arm to RGBA from an arbitrary camera, for compositing.

Uses an EXACT fixed camera built into the MuJoCo model from the requested
:class:`Camera` (intrinsics -> fovy, extrinsic -> pos+quat). This reproduces a
real pinhole faithfully (unlike MuJoCo's free camera, which has no roll and only
azimuth/elevation). The model+camera are cached and rebuilt only when the camera
changes (the third-person viewpoint is fixed, so this is built once).

MuJoCo needs STL/OBJ meshes, so render with ``piper_description.urdf`` (STL), not
the ``_v100_camera`` variant (DAE, which MuJoCo can't decode).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..splat.camera import Camera
from ._external import ensure_on_path


def _camera_quat_wxyz(c2w: np.ndarray):
    """OpenCV cam->world rotation -> MuJoCo camera quat (looks -z, +y up)."""
    from scipy.spatial.transform import Rotation as R
    R_mj = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    qx = R.from_matrix(R_mj).as_quat()      # xyzw
    return [qx[3], qx[0], qx[1], qx[2]]     # wxyz


def build_mujoco_with_camera(urdf: Path, urdf_dir: Path, camera: Camera):
    """Compile a MuJoCo model with a fixed exact camera named ``o2m_cam``."""
    import mujoco
    ensure_on_path()
    from cross_embodiment.mujoco_scene import load_urdf_with_assets

    xml, assets = load_urdf_with_assets(Path(urdf_dir), Path(urdf))
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), camera.width)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), camera.height)
    cam = spec.worldbody.add_camera()
    cam.name = "o2m_cam"
    cam.pos = camera.c2w[:3, 3]
    cam.quat = _camera_quat_wxyz(camera.c2w)
    cam.fovy = camera.fovy_deg
    return spec.compile()


class RobotRenderer:
    def __init__(self, render_urdf: str | Path, urdf_dir: str | Path, joint_prefix: str = ""):
        import mujoco
        ensure_on_path()
        self._mj = mujoco
        self.render_urdf = Path(render_urdf)
        self.urdf_dir = Path(urdf_dir)
        self.joint_prefix = joint_prefix
        self._cache = {}  # camera-pose key -> (model, data, renderer, addrs)

    def _for_camera(self, camera: Camera):
        from cross_embodiment.mujoco_scene import build_joint_index
        key = (round(float(camera.fx), 3), camera.width, camera.height,
               tuple(np.round(camera.c2w.ravel(), 5)))
        if key not in self._cache:
            model = build_mujoco_with_camera(self.render_urdf, self.urdf_dir, camera)
            data = self._mj.MjData(model)
            renderer = self._mj.Renderer(model, height=camera.height, width=camera.width)
            addrs = [a for _, a in sorted(build_joint_index(model, self.joint_prefix).items(),
                                          key=lambda kv: kv[1])]
            self._cache.clear()  # only keep the active camera (fixed viewpoint)
            self._cache[key] = (model, data, renderer, addrs)
        return self._cache[key]

    def render_rgba(self, q: np.ndarray, camera: Camera
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb uint8 HxWx3, alpha bool HxW, depth float HxW)."""
        mj = self._mj
        model, data, renderer, addrs = self._for_camera(camera)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        qv = np.asarray(q, float).reshape(-1)
        for adr, val in zip(addrs, qv):
            data.qpos[int(adr)] = float(val)
        mj.mj_forward(model, data)

        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera="o2m_cam")
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="o2m_cam")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="o2m_cam")
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        alpha = seg[..., 0] >= 0
        return rgb, alpha, depth
