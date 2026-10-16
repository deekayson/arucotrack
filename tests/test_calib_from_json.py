"""`Calib.from_json` is the one way a calibration enters the pipeline: no default file, no dataset lookup, and no
cache posed with another camera's calibration."""
import hashlib
import json

import numpy as np
import pytest

from arucotrack.calibration.camera import Calib, check_calib_for_cache
from arucotrack.geometry import pose as G
from arucotrack.refinement import refine_cache as RC
from tests.conftest import MARKER_M, calib_doc
from tests.conftest import refined_cache_stub as _cache

SQUARE = np.array([[1000, 900], [1040, 900], [1040, 940], [1000, 940]], np.float32)


def _write(tmp_path, doc, name="c.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


def test_from_json_reads_a_schema_2_file_with_its_camera_and_sha(tmp_path):
    p = _write(tmp_path, calib_doc("cam_overhead"))
    c = Calib.from_json(p, marker_m=MARKER_M)
    assert c.camera == "cam_overhead" and c.model == "pinhole" and c.has_extrinsics
    assert c.sha256 == hashlib.sha256(p.read_bytes()).hexdigest() and c.source == str(p)
    assert c.marker_m == MARKER_M
    assert Calib.from_json(p, camera="other", marker_m=0.2).camera == "other"
    assert Calib.from_json(p, marker_m=0.2).marker_m == 0.2


def test_from_json_accepts_a_fisheye_and_refuses_a_file_without_image_size(tmp_path):
    d = calib_doc("cam_fisheye")
    assert Calib.from_json(_write(tmp_path, d, "f.json"), marker_m=MARKER_M).fisheye
    d = calib_doc("cam_overhead")
    del d["image_size"]
    with pytest.raises(ValueError, match="usable"):
        Calib.from_json(_write(tmp_path, d, "n.json"), marker_m=MARKER_M)


def test_there_is_no_default_marker_size(tmp_path):
    c = Calib.from_json(_write(tmp_path, calib_doc("cam_overhead")))
    assert c.marker_m is None
    with pytest.raises(ValueError, match="no marker size"):
        c.obj()


def test_a_cache_is_never_posed_with_another_camera_s_calibration(tmp_path):
    overhead = Calib.from_json(_write(tmp_path, calib_doc("cam_overhead"), "b.json"), marker_m=MARKER_M)
    cache = _cache(tmp_path / "d.npz", "cam_fisheye", np.repeat(SQUARE[None], 3, 0))
    with pytest.raises(ValueError, match="cam_fisheye cache posed with the cam_overhead"):
        G.derive(cache, overhead, out=tmp_path / "g.npz")
    check_calib_for_cache(cache, Calib.from_json(_write(tmp_path, {**calib_doc("cam_overhead"), "camera_id": ""},
                                                            "unnamed.json"), marker_m=MARKER_M))   # names no camera
    unnamed = _cache(tmp_path / "u.npz", "", np.repeat(SQUARE[None], 3, 0))
    check_calib_for_cache(unnamed, overhead)                                  # a cache naming none: nothing either
    fisheye = Calib.from_json(_write(tmp_path, calib_doc("cam_fisheye"), "f.json"), marker_m=MARKER_M)
    pinhole_cache = _cache(tmp_path / "b.npz", "cam_overhead", np.repeat(SQUARE[None], 3, 0))
    with pytest.raises(ValueError, match="cam_overhead cache posed with the cam_fisheye"):
        check_calib_for_cache(pinhole_cache, fisheye)


def test_refine_cache_refuses_another_camera_s_calibration_before_the_video_is_touched(tmp_path, monkeypatch):
    overhead = Calib.from_json(_write(tmp_path, calib_doc("cam_overhead"), "b.json"), marker_m=MARKER_M)
    cache = _cache(tmp_path / "d.npz", "cam_fisheye", np.repeat(SQUARE[None], 3, 0))
    monkeypatch.setattr(RC, "_source", lambda *a, **k: pytest.fail("_source reached"))
    with pytest.raises(ValueError, match="cam_fisheye cache posed with the cam_overhead"):
        RC.run(cache, calib=overhead, out=tmp_path / "r.npz", border_cells=1.0, video=tmp_path / "v.mp4")
