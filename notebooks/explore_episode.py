# %% [markdown]
# # Explore an episode
# Sanity-check raw data: frame counts, EE trajectory, joint ranges vs limits.
# Run as a script or open as a notebook (jupytext `# %%` cells).

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from o2m.config import Config, EpisodePaths
from o2m.data import Episode, load_ee_trajectory, load_joint_trajectory

cfg = Config.from_yaml("../configs/pipeline.yaml")
paths = EpisodePaths.from_config(cfg, "episode_000")

ep = Episode(paths.raw_episode,
             wrist_dir=cfg.get("dataset.cameras.wrist"),
             zed_dir=cfg.get("dataset.cameras.zed"))
df = ep.actions_df()
print("frames:", len(df), "| wrist:", len(ep.wrist_frames()), "| zed:", len(ep.zed_frames()))

# %%
traj = load_ee_trajectory(df, arm="slave", source="ee")
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(*traj.positions.T)
ax.set_title("Slave EE path (base frame, m)")
plt.show()

# %%
joints = load_joint_trajectory(df)
plt.figure(figsize=(8, 3))
plt.plot(joints)
plt.title("Slave joints (rad)")
plt.xlabel("frame")
plt.legend([f"j{i+1}" for i in range(6)], ncol=6, fontsize=7)
plt.show()
print("joint range (rad):", joints.min(), joints.max())

# %%
# Peek at the two camera views.
import cv2
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(cv2.cvtColor(cv2.imread(str(ep.wrist_frames()[0])), cv2.COLOR_BGR2RGB))
axes[0].set_title("wrist (moving)")
axes[1].imshow(cv2.cvtColor(cv2.imread(str(ep.zed_frames()[0])), cv2.COLOR_BGR2RGB))
axes[1].set_title("zed (stationary)")
for a in axes:
    a.axis("off")
plt.show()
