"""The calibration file (`arucotrack.calibration.calibration_file`, `docs/CALIBRATION_FILE.md`): checks on load, the K / R forms,
and camera-frame trajectories when the file has no extrinsics."""
import json

import cv2
import numpy as np
import pytest

from arucotrack.calibration import calibration_file as CF
from arucotrack.calibration.camera import Calib
from arucotrack import pipeline as P
from tests.test_pipeline import _cache, _walk
from tests.conftest import MARKER_M, read_trajectory

K = np.array([[638.0, 0, 1296.0], [0, 638.0, 1035.0], [0, 0, 1.0]])
DIST = [-0.28, 0.18, 0.0002, -0.0006, -0.0067]
RVEC = [-3.1373, 0.0545, -0.0001]
TVEC = [0.0013, -1.3312, 3.1069]


def _doc(**over):
    d = {"camera_id": "cam", "model": "FULL_OPENCV", "image_size": [2592, 1944],
         "intrinsics": {"fx": 638.0, "fy": 638.0, "cx": 1296.0, "cy": 1035.0, "dist": DIST},
         "extrinsics": {"rvec": RVEC, "tvec": TVEC}}
    d.update(over)
    return d


def _write(tmp_path, doc, name="calibration.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_a_good_file_has_no_problems_and_loads():
    assert CF.check(_doc()) == []


@pytest.mark.parametrize("over, match", [
    ({"model": "spherical"}, "model"),
    ({"image_size": [2592]}, "image_size"),
    ({"intrinsics": {"fx": -1, "fy": 638.0, "cx": 1, "cy": 1, "dist": DIST}}, "positive"),
    ({"intrinsics": {"fx": 638.0, "fy": 638.0, "cx": 1, "cy": 1, "dist": [0.1, 0.2, 0.3]}}, "3 coefficients"),
    ({"model": "OPENCV_FISHEYE"}, "expected 4"),
    ({"extrinsics": {"R": np.diag([1.0, 1.0, -1.0]).tolist(), "t": [0, 0, 3]}}, "not a rotation"),
    ({"extrinsics": {"rvec": RVEC}}, "tvec"),
    ({"intrinsics": {"dist": DIST}}, "fx, fy, cx, cy"),
])
def test_problems_are_named_and_refused_on_load(tmp_path, over, match):
    assert any(match in p for p in CF.check(_doc(**over))), CF.check(_doc(**over))
    with pytest.raises(ValueError, match=match):
        Calib.from_json(_write(tmp_path, _doc(**over)), marker_m=MARKER_M)


def test_the_model_must_be_stated_and_is_never_read_from_the_directory(tmp_path):
    d = _doc(); del d["model"]
    assert any("model is missing" in p for p in CF.check(d))
    with pytest.raises(ValueError, match="model is missing"):
        Calib.from_json(_write(tmp_path, d), marker_m=MARKER_M)
    folder = tmp_path / "fisheye_cam"
    folder.mkdir()
    c = Calib.from_json(_write(folder, _doc()), marker_m=MARKER_M)      # a pinhole file in a "fisheye" folder
    assert c.model == "pinhole" and not c.fisheye


def test_the_K_matrix_and_R_t_forms_give_the_same_camera(tmp_path):
    a = Calib.from_json(_write(tmp_path, _doc(), "a.json"), marker_m=MARKER_M)
    R = cv2.Rodrigues(np.asarray(RVEC))[0]
    alt = _doc(intrinsics={"K": K.tolist(), "dist": DIST}, extrinsics={"R": R.tolist(), "t": TVEC})
    b = Calib.from_json(_write(tmp_path, alt, "b.json"), marker_m=MARKER_M)
    assert np.allclose(a.K, b.K) and np.allclose(a.rvec, b.rvec) and np.allclose(a.tvec, b.tvec)
    assert np.allclose(a.cam, b.cam) and a.has_extrinsics and b.has_extrinsics
    X = np.array([[0.4, -0.3, 1.7]])
    assert np.allclose(a.project(X), b.project(X))


def _no_pose_calib(tmp_path):
    return Calib.from_json(_write(tmp_path, _doc(extrinsics=None), "noext.json"), marker_m=MARKER_M)


def test_without_extrinsics_trajectories_are_written_in_the_camera_frame(tmp_path):
    calib = _no_pose_calib(tmp_path)
    assert not calib.has_extrinsics
    # heads 1.4 m in front of the camera, along its optical axis (camera frame: z = depth)
    rows = [(mid, f, np.array([x0, -0.2 + 0.004 * f, 1.4])) for mid, x0 in ((17, 0.1), (23, -0.3)) for f in range(80)]
    c = _cache(tmp_path / "noext.npz", calib, rows)
    no_gates = P.Params(z_min=None, z_max=None, height_min=None, height_max=None)
    out = tmp_path / "out"
    P.metric_step("clip", c, c, out, tmp_path / "work", calib=calib, floor=None, params=no_gates,
                  camera="cam")
    t = read_trajectory(out / "clip.fused.txt")
    assert len(t["id"]) == 160 and np.allclose(t["z"], 1.4, atol=0.02)
    head = (out / "clip.fused.txt").read_text()
    assert "CAMERA frame" in head and "world frame" not in head.split("CAMERA frame")[0]
    assert json.loads((out / "clip.sidecar.json").read_text())["frame"] == "camera"


@pytest.mark.parametrize("floor, params", [
    (P.FLAT, P.Params(z_min=None, z_max=None, height_min=None, height_max=None)),
    (None, P.Params()),
])
def test_without_extrinsics_a_floor_or_a_world_gate_is_refused(tmp_path, floor, params):
    calib = _no_pose_calib(tmp_path)
    c = _cache(tmp_path / "x.npz", calib, _walk(n=20, gap=(0, 0)))
    with pytest.raises(ValueError, match="no extrinsics"):
        P.metric_step("clip", c, c, tmp_path / "o", tmp_path / "w", calib=calib, floor=floor, params=params)


@pytest.mark.parametrize("size", [[2592.5, 1944], [0, 1944], [70000, 1944], [True, 1944], ["2592", 1944]])
def test_image_size_must_be_two_integers_in_range(size):
    assert any("image_size" in p for p in CF.check(_doc(image_size=size)))


def test_coefficients_the_projection_cannot_invert_are_refused():
    """12 or 14 coefficients load only when the thin-prism and tilt terms are zero (`projection.undistort_newton`
    inverts k1..k6, p1, p2)."""
    eight = DIST + [-0.28, 0.18, -0.004]
    ok = _doc(intrinsics={"fx": 638.0, "fy": 638.0, "cx": 1296.0, "cy": 1035.0, "dist": eight + [0.0] * 6})
    assert CF.check(ok) == []
    bad = _doc(intrinsics={"fx": 638.0, "fy": 638.0, "cx": 1296.0, "cy": 1035.0, "dist": eight + [1e-3, 0, 0, 0]})
    assert any("thin-prism" in p for p in CF.check(bad))
    assert any("model" in p for p in CF.check(_doc(model="thin_prism")))
