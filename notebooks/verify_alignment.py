# %% [markdown]
# # Verify reconstruction & alignment
# Plot the COLMAP camera path and check the splat↔base sim3 residual.
# Run after stages 03 (colmap) and 04b (align).

# %%
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from o2m.config import Config, EpisodePaths
from o2m.align import Sim3
from o2m.splat.camera import Camera

cfg = Config.from_yaml("../configs/pipeline.yaml")
paths = EpisodePaths.from_config(cfg, "episode_000")

transforms = json.loads(paths.transforms_json.read_text())
centers = np.array([
    Camera.from_transforms_frame({**transforms, "frames": [fr]}, 0).c2w[:3, 3]
    for fr in transforms["frames"]
])

# %%
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(*centers.T, "-o", ms=2)
ax.set_title("COLMAP wrist-camera path (splat frame)")
plt.show()

# %%
sim3 = Sim3.from_json(paths.sim3_json)
print("sim3 scale:", sim3.s)
print("sim3 t:", sim3.t)
# A small, smooth camera path and a physically plausible scale (table ~0.6 m)
# indicate a healthy reconstruction + alignment.
