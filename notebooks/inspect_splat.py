# %% [markdown]
# # Inspect the trained splat
# Render the splat from a few viewpoints and (optionally) overlay the arm.
# Run after stage 04 (train). Requires the splat + robot extras.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import json
import matplotlib.pyplot as plt
import numpy as np

from o2m.config import Config, EpisodePaths
from o2m.splat import SplatModel
from o2m.splat.camera import Camera, orbit_cameras

cfg = Config.from_yaml("../configs/pipeline.yaml")
paths = EpisodePaths.from_config(cfg, "episode_000")
transforms = json.loads(paths.transforms_json.read_text())

config_yml = sorted(paths.splat.rglob("config.yml"))[-1]
splat = SplatModel.from_config(config_yml)

# %%
base = Camera.from_transforms_frame(transforms, 0)
center = np.mean([
    Camera.from_transforms_frame({**transforms, "frames": [fr]}, 0).c2w[:3, 3]
    for fr in transforms["frames"]], axis=0)
cams = orbit_cameras(base, center, radius=0.6, n=4)

fig, axes = plt.subplots(1, len(cams), figsize=(4 * len(cams), 4))
for ax, cam in zip(np.atleast_1d(axes), cams):
    rgb, _, _ = splat.render(cam)
    ax.imshow(rgb)
    ax.axis("off")
plt.suptitle("Environment splat — orbit views")
plt.show()
