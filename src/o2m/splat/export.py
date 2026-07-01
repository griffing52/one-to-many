"""Export a trained nerfstudio splat to a portable .ply via ``ns-export``."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..utils import get_logger

log = get_logger("o2m.splat")


def export_ply(config_yml: Path, output_dir: Path,
               ns_export_binary: str = "ns-export") -> Path:
    config_yml = Path(config_yml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from .train import resolve_console_script, subprocess_env
    cmd = [resolve_console_script(ns_export_binary), "gaussian-splat",
           "--load-config", str(config_yml),
           "--output-dir", str(output_dir)]
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=subprocess_env())
    plys = sorted(output_dir.glob("*.ply"))
    if not plys:
        raise RuntimeError(f"ns-export produced no .ply in {output_dir}")
    target = output_dir / "splat.ply"
    if plys[0] != target:
        plys[0].replace(target)
    return target
