"""Golden values of `refine.refine` on two fixed synthetic crops: a refactor of the refiner must reproduce them
(1e-6 px). Regenerate only for a deliberate change of the refinement, and say so in the commit."""
import cv2
import numpy as np
import pytest
from arucotrack.refinement import refine

from .conftest import _calib

GOLDEN = {   # crop pixels, the refiner's pixel-EDGE convention, recorded from the refiner's output
    "upright": {"corners": [[31.999999999999886, 31.999999999999886], [192.0, 31.999999999999886], [192.0, 192.0],
                            [32.0, 192.0]], "edge_rms_px": 8.490702752700548e-06, "shift_px": 0.7041521203250414},
    "rotated_blurred": {"corners": [[49.732593684491576, 79.13604515563497], [145.39694667178105, 49.87136916878035],
                                    [174.57099727556806, 145.49323014560923], [79.01059729177882, 174.7415917695123]],
                        "edge_rms_px": 0.015218584930435495, "shift_px": 1.197419775945682},
}


def _crops():
    calib = _calib([[750.63, 0, 1610.5], [0, 749.20, 1337.5], [0, 0, 1]], np.zeros(5), (3222, 2676), "pinhole")   # D = 0
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    out = {}
    # 1: upright 160 px marker 37, crisp
    side, x0, y0 = 160, 32, 32
    img = np.full((224, 224), 200, np.uint8)
    img[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(dic, 37, side)
    truth = np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]], np.float64)
    out["upright"] = (img, truth + [[0.7, -0.4], [-0.3, 0.6], [0.5, 0.5], [-0.6, -0.2]], 37, calib, np.array([900.0, 700.0]))
    # 2: marker 52 rotated 17 deg and blurred, seed off by ~1 px
    big = np.full((448, 448), 200, np.uint8)
    big[124:124 + 200, 124:124 + 200] = cv2.aruco.generateImageMarker(dic, 52, 200)
    M = cv2.getRotationMatrix2D((224.0, 224.0), 17.0, 0.5)
    img2 = cv2.GaussianBlur(cv2.warpAffine(big, M, (448, 448), flags=cv2.INTER_CUBIC, borderValue=200), (5, 5), 1.2)
    img2 = np.ascontiguousarray(img2[112:112 + 224, 112:112 + 224])
    sq = np.array([[124, 124], [324, 124], [324, 324], [124, 324]], np.float64)
    t2 = (np.column_stack([sq, np.ones(4)]) @ M.T) - 112.0
    out["rotated_blurred"] = (img2, t2 + [[1.0, 0.3], [-0.8, 0.9], [0.4, -1.1], [-0.9, -0.5]], 52, calib,
                              np.array([1400.0, 1100.0]))
    return out


def _run(crop, seed, mid, calib, origin):
    return refine.refine(crop, seed, mid, calib, origin=origin, border_cells=1.0)


@pytest.mark.parametrize("name", list(GOLDEN))
def test_refine_reproduces_its_golden_corners(name):
    crop, seed, mid, calib, origin = _crops()[name]
    r = _run(crop, seed, mid, calib, origin)
    want = GOLDEN[name]
    assert r.converged
    assert np.abs(np.asarray(r.corners) - np.asarray(want["corners"])).max() < 1e-6
    assert r.edge_rms_px == pytest.approx(want["edge_rms_px"], abs=1e-6)
    assert r.shift_px == pytest.approx(want["shift_px"], abs=1e-6)
