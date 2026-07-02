# Data

## Raw episode layout

```
pick_bag_joe/
  episode_000/
    actions.csv
    realsense_color/000000.png ...   # wrist camera (RealSense D435i), 640x480
    zed_color/000000.png ...         # stationary third-person (ZED2), 1280x720
  episode_001/ ...
```

60 episodes, ~350–550 frames each. Camera streams are frame-aligned with the CSV
rows. **No camera intrinsics/extrinsics and no depth** are stored.

## `actions.csv` columns (36)

| Group | Columns | Units |
|-------|---------|-------|
| Frame | `frame`, `timestamp` | index, UNIX s |
| Master joints | `master_j1..j6`, `master_gripper_angle`, `master_gripper_effort` | 0.001 deg / 0.001 mm |
| Master EE | `master_ee_x/y/z`, `master_ee_rx/ry/rz` | 0.001 mm / 0.001 deg |
| Slave joints | `slave_j1..j6`, `slave_gripper_angle`, `slave_gripper_effort` | 0.001 deg / 0.001 mm |
| Slave EE | `slave_ee_x/y/z`, `slave_ee_rx/ry/rz` | 0.001 mm / 0.001 deg |
| Slave FK | `slave_fk_x/y/z`, `slave_fk_rx/ry/rz` | mm / deg |

Training and deployment use the **slave** arm.

## Unit conventions (loaders)

`o2m.data.actions` reproduces the LeRobot converter exactly:

- position: `0.001 mm → m` (× `1e-6`);
- rotation: `0.001 deg → rad` (× `π/180000`), Euler XYZ **unwrapped**, then
  scipy `xyz` → rotation vector;
- gripper: `abs(value) × 1e-6 → m`.

```python
from o2m.data import Episode, load_ee_trajectory, load_joint_trajectory
ep = Episode(".../pick_bag_joe/episode_000")
traj = load_ee_trajectory(ep.actions_df(), arm="slave", source="ee")  # base frame, SI
joints = load_joint_trajectory(ep.actions_df())                        # (N, 6) rad
```

We do **not** convert to LeRobot for splatting — raw full-resolution PNGs are
preferred. The LeRobot converter remains useful for policy training.

!!! note "Joint units"
    `slave_j*` are recorded in 0.001 deg like the EE rotations; verify against
    URDF joint limits at runtime (a quick sanity check lives in
    `notebooks/explore_episode.py`).
