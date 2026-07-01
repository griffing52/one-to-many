"""Train a splat with nerfstudio ``splatfacto``.

Shells out to ``ns-train`` so nerfstudio's CLI/config machinery stays intact,
then locates the produced config.yml. Auto-orient / auto-scale are disabled so
the trained splat lives in the COLMAP world frame (required by the sim3
alignment used for the robot overlay).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..utils import get_logger

log = get_logger("o2m.splat")


def resolve_console_script(name: str) -> str:
    """Find a console script (ns-train/ns-export) even when the env's bin is not
    on PATH: prefer PATH, else look next to the running interpreter."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return str(candidate)
    return name  # last resort; subprocess will raise a clear error


def subprocess_env() -> dict:
    """Env for nerfstudio subprocesses: ensure the interpreter's bin dir (with
    `ninja`, needed by gsplat's CUDA JIT) is on PATH, and CUDA_HOME is set."""
    import os

    env = dict(os.environ)
    bindir = str(Path(sys.executable).parent)
    if bindir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    if not env.get("CUDA_HOME"):
        nvcc = shutil.which("nvcc", path=env["PATH"])
        if nvcc:
            env["CUDA_HOME"] = str(Path(nvcc).parent.parent)
    # gsplat JIT-compiles its CUDA kernels (no prebuilt wheel for this torch/CUDA
    # combo). Each `cicc` compiler uses ~3 GB RAM; compiling many in parallel
    # OOMs a shared box. Cap parallel build jobs so RAM stays bounded. Kernels
    # are cached after the first build, so this cost is paid once.
    env.setdefault("MAX_JOBS", "2")
    return env


def ensure_build_env_in_process() -> None:
    """Apply the same PATH/CUDA_HOME/MAX_JOBS setup to THIS process's os.environ,
    so in-process gsplat JIT compilation (e.g. at render time) can find `ninja`
    and `nvcc`. Kernels are cached after the first build."""
    import os

    for key, val in subprocess_env().items():
        if key in ("PATH", "CUDA_HOME", "MAX_JOBS"):
            os.environ[key] = val


class SplatTrainer:
    def __init__(self, method: str = "splatfacto", ns_train_binary: str = "ns-train",
                 max_num_iterations: int = 30000,
                 orientation_method: str = "none", center_method: str = "none",
                 auto_scale_poses: bool = False,
                 cache_images: str = "cpu", cache_images_type: str = "uint8"):
        self.method = method
        self.binary = ns_train_binary
        self.max_num_iterations = max_num_iterations
        self.orientation_method = orientation_method
        self.center_method = center_method
        self.auto_scale_poses = auto_scale_poses
        # Keep memory modest (this box co-runs other GPU/RAM-heavy jobs): cache
        # images on CPU as uint8 rather than the GPU-float default.
        self.cache_images = cache_images
        self.cache_images_type = cache_images_type

    def train(self, nerfstudio_dir: Path, output_dir: Path,
              max_iters: Optional[int] = None) -> Path:
        """Run training; return the path to the produced ``config.yml``."""
        nerfstudio_dir = Path(nerfstudio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        iters = max_iters or self.max_num_iterations

        cmd = [
            resolve_console_script(self.binary), self.method,
            "--data", str(nerfstudio_dir),
            "--output-dir", str(output_dir),
            "--max-num-iterations", str(iters),
            "--viewer.quit-on-train-completion", "True",
            # Memory-safety: cache images on CPU as uint8 (pipeline args must
            # precede the `nerfstudio-data` dataparser subcommand).
            "--pipeline.datamanager.cache-images", self.cache_images,
            "--pipeline.datamanager.cache-images-type", self.cache_images_type,
            # CRITICAL: keep the splat in the COLMAP frame.
            "nerfstudio-data",
            "--orientation-method", self.orientation_method,
            "--center-method", self.center_method,
            "--auto-scale-poses", str(self.auto_scale_poses),
        ]
        log.info("$ %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=subprocess_env())

        configs = sorted(output_dir.rglob("config.yml"),
                         key=lambda p: p.stat().st_mtime)
        if not configs:
            raise RuntimeError(f"No config.yml found under {output_dir} after training")
        return configs[-1]
