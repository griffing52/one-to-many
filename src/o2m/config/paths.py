"""Single source of truth for every artifact path under outputs/<episode>/.

Mirrors the ``cross_embodiment/paths.py`` style: one object that derives all
downstream paths so scripts never hand-build strings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EpisodePaths:
    episode_id: str
    output_root: Path
    data_root: Path

    @classmethod
    def from_config(cls, cfg, episode_id: str | None = None) -> "EpisodePaths":
        episode_id = episode_id or cfg.require("episode")
        return cls(
            episode_id=episode_id,
            output_root=Path(cfg.get("output_root", "./outputs")).resolve(),
            data_root=Path(cfg.require("dataset.data_root")).resolve(),
        )

    # --- raw input ---------------------------------------------------------
    @property
    def raw_episode(self) -> Path:
        return self.data_root / self.episode_id

    def raw_camera(self, dir_name: str) -> Path:
        return self.raw_episode / dir_name

    @property
    def actions_csv(self) -> Path:
        return self.raw_episode / "actions.csv"

    # --- per-episode output tree -------------------------------------------
    @property
    def root(self) -> Path:
        return self.output_root / self.episode_id

    @property
    def frames(self) -> Path:
        return self.root / "frames"          # COLMAP `images/`

    @property
    def masks(self) -> Path:
        return self.root / "masks"

    @property
    def colmap(self) -> Path:
        return self.root / "colmap"

    @property
    def colmap_sparse(self) -> Path:
        return self.colmap / "sparse" / "0"

    @property
    def colmap_db(self) -> Path:
        return self.colmap / "database.db"

    @property
    def nerfstudio(self) -> Path:
        return self.root / "nerfstudio"

    @property
    def transforms_json(self) -> Path:
        return self.nerfstudio / "transforms.json"

    @property
    def splat(self) -> Path:
        return self.root / "splat"

    @property
    def splat_ply(self) -> Path:
        return self.splat / "splat.ply"

    @property
    def align(self) -> Path:
        return self.root / "align"

    @property
    def sim3_json(self) -> Path:
        return self.align / "sim3.json"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    def render_dir(self, mode: str, edit: str) -> Path:
        return self.renders / mode / edit

    def ensure(self) -> "EpisodePaths":
        for p in (self.frames, self.masks, self.colmap, self.nerfstudio,
                  self.splat, self.align, self.renders):
            p.mkdir(parents=True, exist_ok=True)
        return self
