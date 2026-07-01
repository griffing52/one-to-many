import numpy as np
import pytest

from o2m.config import Config, EpisodePaths
from o2m.data.types import EETrajectory
from o2m.trajectory import apply_preset, PRESETS, TrajectoryEditor


CONFIG = "configs/pipeline.yaml"


def test_config_includes_merge():
    cfg = Config.from_yaml(CONFIG)
    # value from an included file
    assert cfg.get("dataset.cameras.wrist") == "realsense_color"
    # value from the top-level file
    assert cfg.get("render.mode") == "robot_overlay"


def test_episode_paths_derivation():
    cfg = Config.from_yaml(CONFIG)
    paths = EpisodePaths.from_config(cfg, "episode_007")
    assert paths.episode_id == "episode_007"
    assert paths.transforms_json.name == "transforms.json"
    assert paths.render_dir("static", "replay_original").parts[-2:] == ("static", "replay_original")


def _toy_traj(n=10):
    rng = np.random.default_rng(1)
    return EETrajectory(np.arange(float(n)), rng.normal(size=(n, 3)),
                        rng.normal(size=(n, 3)) * 0.1, np.ones(n) * 0.03)


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_run(name):
    out = apply_preset(name, _toy_traj())
    assert out.positions.shape[1] == 3
    assert len(out) == len(out.rotvecs)


def test_time_warp_changes_length():
    out = TrajectoryEditor(_toy_traj(10)).time_warp(1.5).result()
    assert len(out) == 15


def test_translate_offsets_positions():
    tr = _toy_traj()
    out = TrajectoryEditor(tr).translate((1.0, 0.0, 0.0)).result()
    assert np.allclose(out.positions[:, 0] - tr.positions[:, 0], 1.0)
