"""The multi-pass union must add recall without adding ambiguity: dedupe, conflicts, scale mapping, bounds."""
import cv2
import numpy as np
import pytest

from arucotrack.refinement import refine
from tests import synthetic
from arucotrack.detection import detectors
from arucotrack.detection.detectors import Detection
from tests.conftest import reference_calib

ROSTER = [17, 18, 19, 20, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44]
CAL = reference_calib()


def _render(mid=37, side=60.0, tilt=12.0, seed=3):
    rng = np.random.default_rng(seed)
    rvec, tvec = synthetic.pose_at(CAL, 0.35, side, tilt, 0.7, rng)
    r = synthetic.render(CAL, mid, rvec, tvec, rng=rng, bg=78.0, margin_cells=0.3, border_cells=1.0)
    return r


def test_roster_bound_is_computed_not_assumed():
    assert detectors.roster_min_distance(ROSTER) == 14
    assert detectors.roster_hmax(ROSTER) == 6
    assert detectors.roster_min_distance(ROSTER) >= 2 * detectors.roster_hmax(ROSTER) + 1
    assert detectors.roster_min_distance(list(range(250))) == 12          # the full dictionary is 36h12


def test_single_pass_multipass_equals_opencv():
    r = _render()
    base = {"adaptiveThreshWinSizeMax": 73, "adaptiveThreshWinSizeStep": 4, "perspectiveRemovePixelPerCell": 8}
    mp = detectors.MultiPass("subpix", base, constants=(6,), scales=(1,))
    cv = detectors.OpenCVAruco("subpix", dict(base, adaptiveThreshConstant=6))
    a, b = mp.detect_with_rejected(r.patch)[0], cv.detect_with_rejected(r.patch)[0]
    assert [d.id for d in a] == [d.id for d in b] == [37]
    assert np.array_equal(a[0].corners, b[0].corners)


def test_union_dedupes_to_one_detection_and_keeps_the_lowest_pass():
    r = _render()
    mp = detectors.MultiPass(constants=(3, 6, 9), scales=(1, 2))
    out = mp.detect_with_rejected(r.patch)[0]
    assert [d.id for d in out] == [37]
    assert out[0].extra["pass"] == (1.0, 3) and out[0].extra["n_passes"] >= 4
    assert mp.conflicts == 0


def test_id_conflict_drops_the_cluster_and_is_counted():
    r = _render()
    mp = detectors.MultiPass(constants=(3, 6), scales=(1,))
    real = mp._passes[1][1].detect_with_rejected

    def liar(gray):
        out, rej = real(gray)
        for d in out:
            d.id = 44
        return out, rej
    mp._passes[1][1].detect_with_rejected = liar
    assert mp.detect_with_rejected(r.patch)[0] == []
    assert mp.conflicts == 1


def test_upscaled_pass_maps_back_to_the_same_corners():
    r = _render(side=45.0)
    one = detectors.MultiPass(constants=(6,), scales=(1,)).detect_with_rejected(r.patch)[0]
    two = detectors.MultiPass(constants=(6,), scales=(2,)).detect_with_rejected(r.patch)[0]
    assert len(one) == len(two) == 1
    raw = np.linalg.norm(one[0].corners - two[0].corners, axis=1).max()
    assert raw < 1.0, raw                                     # same marker, subpix on a cubic upscale differs a little
    # "the seed does not matter": after the interior refiner the two seeds agree far better
    ra = refine.refine(r.patch, one[0].corners, 37, CAL, origin=r.origin, border_cells=1.0)
    rb = refine.refine(r.patch, two[0].corners, 37, CAL, origin=r.origin, border_cells=1.0)
    assert ra.converged and rb.converged
    assert np.linalg.norm(ra.corners - rb.corners, axis=1).max() < 0.05
    # and the half-pixel convention is the right one: the plain x/k mapping would put the 2x corners
    # 0.25 px off in both axes on average (measured +0.28/+0.25 over 60 renders, half-pixel: +0.03/+0.00)
    offs = []
    for seed in range(12):
        r = _render(side=55.0 + 3 * seed, tilt=5.0 + 2 * seed, seed=seed)
        if r is None:
            continue
        a = detectors.MultiPass(constants=(6,), scales=(1,)).detect_with_rejected(r.patch)[0]
        b = detectors.MultiPass(constants=(6,), scales=(2,)).detect_with_rejected(r.patch)[0]
        if len(a) == 1 and len(b) == 1:
            offs.append((b[0].corners - a[0].corners).mean(0))
    assert len(offs) >= 8
    assert np.abs(np.mean(offs, 0)).max() < 0.1, np.mean(offs, 0)


def test_roster_multipass_uses_the_six_bit_bound_and_rejects_off_roster():
    r = _render(mid=37)
    mp = detectors.MultiPass(constants=(3, 6, 9), scales=(1,), roster=ROSTER)
    assert mp.hmax == 6
    assert [d.id for d in mp.detect_with_rejected(r.patch)[0]] == [37]
    off = _render(mid=65)
    assert mp.detect_with_rejected(off.patch)[0] == []
    assert mp.detect_with_rejected(off.patch)[1].shape[1:] == (4, 2)


def test_parse_spec_builds_multipass():
    mp = detectors.parse_spec("multipass:subpix ac=3,6,9 scales=1,2 iou=0.4 roster adaptiveThreshWinSizeMax=73", ROSTER)
    assert isinstance(mp, detectors.MultiPass)
    assert mp.constants == (3, 6, 9) and mp.scales == (1.0, 2.0) and mp.roster == ROSTER and mp.iou == 0.4
    assert mp.params["adaptiveThreshWinSizeMax"] == 73 and mp.params["perspectiveRemovePixelPerCell"] == 8
    with pytest.raises(ValueError, match="kind"):
        detectors.parse_spec("opencv:subpix adaptiveThreshConstant=6")
    with pytest.raises(ValueError):
        detectors.parse_spec("multipass: roster", None)
