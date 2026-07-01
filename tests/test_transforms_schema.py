import numpy as np

from o2m.colmap.model_io import ColmapModel, ColmapCamera, ColmapImage
from o2m.colmap.to_nerfstudio import colmap_to_transforms


def _toy_model():
    cam = ColmapCamera("OPENCV", 640, 480,
                       np.array([600, 600, 320, 240, 0.01, -0.002, 0.0, 0.0]))
    images = {
        1: ColmapImage("000000.png", np.array([1.0, 0, 0, 0]), np.array([0.1, 0.2, 0.3]), 0),
        2: ColmapImage("000001.png", np.array([0.9239, 0, 0.3827, 0]), np.array([0, 0, 1.0]), 0),
    }
    return ColmapModel({0: cam}, images, np.zeros((0, 3)), np.zeros((0, 3)))


def test_transforms_top_level_intrinsics():
    t = colmap_to_transforms(_toy_model())
    assert t["camera_model"] == "OPENCV"
    assert (t["w"], t["h"]) == (640, 480)
    assert t["fl_x"] == 600 and t["cx"] == 320
    assert t["k1"] == 0.01 and t["p1"] == 0.0


def test_transforms_frames_and_masks():
    t = colmap_to_transforms(_toy_model(), mask_subdir="masks")
    assert len(t["frames"]) == 2
    fr = t["frames"][0]
    assert fr["file_path"] == "images/000000.png"
    assert fr["mask_path"] == "masks/000000.png.png"
    M = np.array(fr["transform_matrix"])
    assert M.shape == (4, 4)
    assert np.allclose(M[3], [0, 0, 0, 1])
