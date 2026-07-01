import numpy as np

from o2m.utils import geometry as g
from o2m.align import WorldAligner, Sim3


def test_opencv_opengl_involution():
    C = g.se3_from_rotvec([0.2, 0.1, -0.3], [1, 2, 3])
    assert np.allclose(g.opengl_c2w_to_opencv(g.opencv_c2w_to_opengl(C)), C)


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(0)
    R = g.se3_from_euler([0.3, -0.2, 0.5], [0, 0, 0])[:3, :3]
    s, t = 2.5, np.array([1.0, -2.0, 0.5])
    base = rng.normal(size=(40, 3))
    splat = (s * (R @ base.T).T) + t

    sim3, diag = WorldAligner.from_wrist_fk(splat, base)
    assert diag["residual_rms"] < 1e-6
    assert abs(sim3.s - s) < 1e-6
    assert np.allclose(sim3.R, R, atol=1e-6)
    assert np.allclose(sim3.t, t, atol=1e-6)


def test_sim3_inverse_round_trip():
    sim3 = Sim3(1.7, g.se3_from_euler([0.1, 0.2, -0.1], [0, 0, 0])[:3, :3],
                np.array([0.5, -0.3, 0.2]))
    P = g.se3_from_rotvec([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert np.allclose(sim3.inv_apply(sim3.apply(P)), P, atol=1e-9)


def test_sim3_json_round_trip(tmp_path):
    sim3 = Sim3(2.0, np.eye(3), np.array([1.0, 2.0, 3.0]))
    p = sim3.to_json(tmp_path / "sim3.json")
    loaded = Sim3.from_json(p)
    assert loaded.s == sim3.s and np.allclose(loaded.t, sim3.t)
