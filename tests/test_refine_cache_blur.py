"""The blur window of the refinement (`refine.blur_window`): the velocity, the exposure fraction (config preferred, else the video) and what the
refined cache records about both."""
from __future__ import annotations

import json

import numpy as np
import pytest

from arucotrack.refinement import exposure_blur
from arucotrack.refinement import refine_cache
from tests.conftest import PinholeCanvas, make_warped_marker, video_from_frames
from tests.test_refine_blur_window import B, CANVAS, blurred_marker

SPEC = "multipass:subpix ac=3,6,9 scales=1"


# ------------------------------------------------------------------ the exposure fraction from the images
@pytest.fixture(scope="module")
def blur_measurements():
    """Known exposure fraction 0.7, optics sigma 0.6 px. Speeds 8-16 px/frame in several directions. The reference
    quad is the true one stretched 3 px along the motion, as a detector's quad on a blurred marker is."""
    f, sig = 0.7, 0.6
    rows = []
    for k, (mid, ang, speed) in enumerate([(18, 0.0, 9.0), (25, 90.0, 12.0), (37, 30.0, 15.0), (40, 160.0, 11.0),
                                           (18, 250.0, 14.0), (33, 300.0, 8.5)]):
        a = np.radians(ang)
        v = speed * np.array([np.cos(a), np.sin(a)])
        img, origin, truth = blurred_marker(mid, 120, v, f, sigma=sig, seed=k)
        u = v / speed
        stretched = truth + 1.5 * np.sign((truth - truth.mean(0)) @ u)[:, None] * u[None, :]
        for vn, w in exposure_blur.blur_widths(img, stretched, mid, B, v, sig):
            rows.append(dict(row=k, marker_id=mid, vn=vn, w=w))
    return f, rows


def test_the_image_estimator_recovers_a_known_exposure_fraction(blur_measurements):
    f, rows = blur_measurements
    assert len(rows) >= 6
    est = exposure_blur.estimate_exposure_fraction(rows, min_rows=5)
    assert est["ok"], est
    assert abs(est["value"] - f) < 0.04, est
    assert est["n_rows"] == len({r["row"] for r in rows if r["vn"] >= exposure_blur.VMIN})
    assert est["sd"] is not None and est["sd"] < 0.05
    assert est["n_markers"] == len({r["marker_id"] for r in rows})


def test_the_image_estimator_refuses_too_few_fast_rows(blur_measurements):
    _, rows = blur_measurements
    est = exposure_blur.estimate_exposure_fraction(rows, min_rows=50)
    assert not est["ok"] and est["value"] is None and "50" in est["reason"]


# ------------------------------------------------------------------ the refined cache: provenance and refusals
@pytest.fixture
def moving_cache(tmp_path, pinhole_calib):
    """A marker crossing the frame at 12 px/frame (4 frames)."""
    r = PinholeCanvas(pinhole_calib)
    W, H = r.size
    frames = [make_warped_marker(r, 37, 120, centre_rect=(W // 2 - 20 + 12 * k, H // 2))[0] for k in range(4)]
    vid = tmp_path / "clip.mp4"
    video_from_frames(vid, frames)
    from arucotrack.detection import detect
    cache = detect.run(str(vid), source="sw_y", tag="blur", cache_root=tmp_path / "cache",
                       stem="clip__cam", det_spec=SPEC, patch=0)["cache"]
    return cache, pinhole_calib


def _run(cache, calib, tmp_path, name, **kw):
    return refine_cache.run(cache, calib=calib, video=tmp_path / "clip.mp4", source="sw_y", border_cells=1.0,
                            out=tmp_path / name, **kw)


def test_a_config_exposure_fraction_is_used_and_recorded_beside_the_video_estimate(tmp_path, moving_cache):
    cache, calib = moving_cache
    z = np.load(_run(cache, calib, tmp_path, "bw.refined.npz", blur_window=True, exposure_fraction=0.9,
                     sigma_optics=0.5), allow_pickle=False)
    prov = json.loads(str(z["provenance"]))
    bw = prov["parameters"]["blur_window"]
    assert bw["enabled"] and bw["sigma_optics_px"] == 0.5
    ex = bw["exposure_fraction"]
    assert ex["source"] == "config" and ex["value"] == 0.9 and ex["config"] == 0.9
    assert ex["video"]["ok"] is False and ex["video"]["reason"]          # 4 rows: no estimate, said so, not refused
    vel = bw["velocity"]
    assert vel["n_central"] + vel["n_one_sided"] + vel["n_none"] == len(z["det_id"])
    assert vel["n_central"] >= 1 and vel["n_one_sided"] >= 2
    assert vel["time"].startswith("det_pts") and abs(vel["nominal_period_s"] - 1 / 30) < 1e-3   # 30 fps clip
    assert prov["summary"]["blur_window"] == bw
    v = z["det_velocity_px"]
    assert v.shape == (len(z["det_id"]), 2) and np.isfinite(v).all()
    assert np.allclose(np.median(v[:, 0]), 12.0, atol=1.5)


def test_without_a_config_value_too_few_fast_rows_are_refused_before_anything_is_written(tmp_path, moving_cache):
    cache, calib = moving_cache
    with pytest.raises(ValueError, match="fast rows"):
        _run(cache, calib, tmp_path, "nope.refined.npz", blur_window=True, sigma_optics=0.5)
    assert not (tmp_path / "nope.refined.npz").exists()


def test_a_video_estimate_is_used_without_a_config_value_and_recorded_with_it(tmp_path, moving_cache, monkeypatch):
    cache, calib = moving_cache
    est = dict(ok=True, value=0.95, sd=0.012, sd_method="bootstrap over markers", n_rows=180, n_measurements=300,
               n_markers=12, reason=None)
    seen = {}

    def fake(*a, **kw):
        seen.update(kw)
        return dict(est)
    monkeypatch.setattr(refine_cache, "video_exposure_estimate", fake)
    z = np.load(_run(cache, calib, tmp_path, "vid.refined.npz", blur_window=True, sigma_optics=0.5), allow_pickle=False)
    ex = json.loads(str(z["provenance"]))["parameters"]["blur_window"]["exposure_fraction"]
    assert ex["source"] == "video" and ex["value"] == 0.95 and ex["config"] is None
    assert ex["video"]["sd"] == 0.012 and ex["video"]["n_rows"] == 180
    assert seen["sigma_optics"] == 0.5
    z = np.load(_run(cache, calib, tmp_path, "both.refined.npz", blur_window=True, sigma_optics=0.5,
                     exposure_fraction=1.0), allow_pickle=False)
    ex = json.loads(str(z["provenance"]))["parameters"]["blur_window"]["exposure_fraction"]
    assert ex["source"] == "config" and ex["value"] == 1.0 and ex["video"]["value"] == 0.95


def test_without_the_blur_window_nothing_changes(tmp_path, moving_cache):
    cache, calib = moving_cache
    a = np.load(_run(cache, calib, tmp_path, "a.refined.npz"), allow_pickle=False)
    assert json.loads(str(a["provenance"]))["parameters"]["blur_window"] == {"enabled": False}
    assert "det_velocity_px" not in a.files
    with pytest.raises(ValueError, match="refine.blur_window"):
        _run(cache, calib, tmp_path, "b.refined.npz", sigma_optics=0.5)
    with pytest.raises(ValueError, match="refine.blur_window"):
        _run(cache, calib, tmp_path, "c.refined.npz", exposure_fraction=1.0)


def test_the_blur_window_needs_the_optics_sigma(tmp_path, moving_cache):
    cache, calib = moving_cache
    with pytest.raises(ValueError, match="sigma"):
        _run(cache, calib, tmp_path, "d.refined.npz", blur_window=True, exposure_fraction=1.0)


# ------------------------------------------------------------------ the marker dictionary reaches the pattern model
def test_blur_widths_models_the_pattern_of_the_given_dictionary(monkeypatch):
    import cv2

    from arucotrack.refinement import refine
    other = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    real, used = refine.cell_grid, []

    def spy(marker_id, dictionary=None):
        g = real(marker_id, dictionary)
        used.append(g)
        return g
    monkeypatch.setattr(refine, "cell_grid", spy)
    img, _, truth = blurred_marker(18, 120, np.array([12.0, 0.0]), 0.7, sigma=0.6, seed=0)
    exposure_blur.blur_widths(img, truth, 18, B, np.array([12.0, 0.0]), 0.6, dictionary=other)
    assert used and np.array_equal(used[0], real(18, other))
    assert not np.array_equal(real(18, other), real(18))           # the two dictionaries differ for this id


def test_the_refinement_passes_its_dictionary_to_the_exposure_estimate(tmp_path, moving_cache, monkeypatch):
    import cv2
    cache, calib = moving_cache
    real, seen = exposure_blur.blur_widths, []

    def spy(*a, dictionary=None, **k):
        seen.append(dictionary)
        return real(*a, dictionary=dictionary, **k)
    monkeypatch.setattr(exposure_blur, "blur_widths", spy)
    _run(cache, calib, tmp_path, "d.refined.npz", blur_window=True, exposure_fraction=0.9, sigma_optics=0.5,
         dictionary="DICT_6X6_250")
    assert seen and all(d is not None and np.array_equal(d.bytesList,
                        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250).bytesList) for d in seen)
