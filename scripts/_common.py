"""Shared argparse / config bootstrap for the stage scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the scripts without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from o2m.config import Config, EpisodePaths  # noqa: E402
from o2m.utils import get_logger  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config.")
    p.add_argument("--episode", default=None, help="Episode id (overrides config).")
    return p


def load(args) -> tuple[Config, EpisodePaths]:
    cfg = Config.from_yaml(args.config)
    paths = EpisodePaths.from_config(cfg, episode_id=args.episode).ensure()
    return cfg, paths


log = get_logger("o2m.scripts")
