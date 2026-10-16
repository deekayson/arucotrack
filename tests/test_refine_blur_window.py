"""Blur-adaptive profile window (`refine.refine(velocity=...)`): exposure motion blur along the marker's travel.

During an exposure the marker moves `exposure_fraction x |v . n|` px across an edge of normal `n`. The edge is then a
ramp that wide. The production half-window (<= 4 px) sits inside such a ramp, so the gradient centroid follows the
seed instead of the edge (the frame-to-frame "edge flips"). With the velocity the window is widened per edge sample to
cover the ramp, capped short of the next same-polarity edge. Without it nothing changes, bit for bit.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from arucotrack.refinement import refine
from tests import synthetic

from .conftest import _calib

# A zero-distortion pinhole camera at the scale of the overhead cameras.
K_CANVAS = [[750.630852, 0.0, 1610.5], [0.0, 749.203421, 1337.5], [0.0, 0.0, 1.0]]
CANVAS = _calib(K_CANVAS, np.zeros(5), (3222, 2676), "pinhole")
B = 0.89151                       # the printed-border constant (any value would do, the renderer and refiner share it)
RVEC = np.array([np.pi, 0.0, 0.0])  # marker facing the camera, its +y up in the image


def _tvec(centre_px, side_px):
    fx, fy, cx, cy = CANVAS.K[0, 0], CANVAS.K[1, 1], CANVAS.K[0, 2], CANVAS.K[1, 2]
    z = fx * CANVAS.marker_m / side_px
    return np.array([(centre_px[0] - cx) * z / fx, (centre_px[1] - cy) * z / fy, z])


def blurred_marker(mid, side_px, v, f, *, sigma=0.5, noise=1.5, seed=0, P=224, centre=(1810.3, 1187.6)):
    """An exposure: `synthetic.render_raw` averaged over the marker's path during the open shutter (`f x v` px, centred
    on the mid-exposure pose), then optics blur and noise. Returns (crop, origin, true outer corners at
    mid-exposure in crop px, refiner edge convention)."""
    v = np.asarray(v, np.float64)
    t0 = _tvec(centre, side_px)
    origin = np.floor(np.asarray(centre) - P / 2)
    ideal = synthetic.subsample_ideal(CANVAS, origin, P, 4)
    blur = f * np.linalg.norm(v)
    n = max(1, int(np.ceil(blur * 4)))
    acc = np.zeros((P, P))
    for s in (np.arange(n) + 0.5) / n - 0.5:       # n equal slices of the path, each at its middle: a box of `blur`
        t = _tvec(np.asarray(centre) + s * f * v, side_px)
        acc += synthetic.render_raw(CANVAS, mid, RVEC, t, origin, P, supersample=4, border_cells=B, bg=170.0,
                                ideal=ideal)
    img = acc / n
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
    truth = synthetic.project_outer(CANVAS, RVEC, t0) - origin
    return np.clip(img, 0, 255), origin, truth


def _err(c, truth):
    """rms corner distance, px (as `test_refine` measures the refiner)."""
    return float(np.sqrt(((np.asarray(c) - truth) ** 2).sum(1).mean()))


# ------------------------------------------------------------------ the bite: seed independence on a blurred marker
BLUR_CASE = dict(mid=18, side_px=120, v=(9.0, 0.0), f=1.0)       # blur 9 px, 0.6 of a cell (side / 8 = 15 px)


@pytest.fixture(scope="module")
def blurred():
    return blurred_marker(BLUR_CASE["mid"], BLUR_CASE["side_px"], BLUR_CASE["v"], BLUR_CASE["f"])


def _refine(img, seed, origin, **kw):
    return refine.refine(img, seed, BLUR_CASE["mid"], CANVAS, origin=origin, border_cells=B, **kw)


@pytest.mark.parametrize("shift", [-3.0, 3.0])
def test_the_blur_window_finds_the_mid_exposure_marker_whatever_the_seed(blurred, shift):
    img, origin, truth = blurred
    seed = truth + np.array([shift, 0.0])                     # seed shifted along the motion
    r = _refine(img, seed, origin, velocity=BLUR_CASE["v"], exposure_fraction=BLUR_CASE["f"], sigma_optics=0.5)
    assert r.converged, r.reason
    assert _err(r.corners, truth) < 0.3, f"seed {shift:+} px: corners {_err(r.corners, truth):.3f} px off mid-exposure"


def test_the_production_window_follows_the_seed_on_the_same_marker(blurred):
    """Without the velocity the window follows the seed: the tolerance the blur window meets is not met."""
    img, origin, truth = blurred
    errs = []
    for shift in (-3.0, 3.0):
        r = _refine(img, truth + np.array([shift, 0.0]), origin)
        assert r.converged, r.reason
        errs.append(_err(r.corners, truth))
    assert max(errs) > 0.3, f"the production window already meets the tolerance ({errs}); the test does not bite"


def test_the_blur_window_result_does_not_depend_on_the_seed(blurred):
    img, origin, truth = blurred
    out = [_refine(img, truth + np.array([s, 0.0]), origin, velocity=BLUR_CASE["v"],
                   exposure_fraction=BLUR_CASE["f"], sigma_optics=0.5).corners for s in (-3.0, 3.0)]
    assert _err(out[0], out[1]) < 0.1


def test_near_a_cell_of_blur_the_position_holds_and_a_size_bias_remains():
    """The known limit (about -1 px of size at a blur of a cell or more, with every method): at 13 px of blur
    (0.85 cell) the mid-exposure position and the seed independence hold, the quad reads up to ~0.6 px small along
    the motion. Recorded so a change of that limit is seen, in either direction."""
    img, origin, truth = blurred_marker(18, 120, (13.0, 0.0), 1.0)
    for shift in (-3.0, 3.0):
        c = refine.refine(img, truth + np.array([shift, 0.0]), 18, CANVAS, origin=origin, border_cells=B,
                          velocity=(13.0, 0.0), exposure_fraction=1.0, sigma_optics=0.5).corners
        e_t = c[:, 0].mean() - truth[:, 0].mean()
        e_s = (c[1, 0] + c[2, 0] - c[0, 0] - c[3, 0]) / 2 - (truth[1, 0] + truth[2, 0] - truth[0, 0] - truth[3, 0]) / 2
        assert abs(e_t) < 0.15, f"seed {shift:+}: translation {e_t:+.3f} px"
        assert -0.8 < e_s < 0.0, f"seed {shift:+}: size along the motion {e_s:+.3f} px"


def test_the_quality_pass_keeps_the_production_window(blurred):
    """`edge_rms_px` / `downweighted_frac` are the production `edge_residuals` at the result (<= 4 px window), so the
    gate keeps its meaning."""
    img, origin, truth = blurred
    r = _refine(img, truth, origin, velocity=BLUR_CASE["v"], exposure_fraction=BLUR_CASE["f"], sigma_optics=0.5)
    d, ok = refine.edge_residuals(img, r.corners, BLUR_CASE["mid"], CANVAS, origin=origin, border_cells=B)
    assert (r.edge_rms_px, r.downweighted_frac) == refine.edge_quality(d, ok)


# ------------------------------------------------------------------ no velocity, or no blur: the plain refiner exactly
def test_a_static_marker_refines_identically_with_the_blur_window_on_or_off():
    img, origin, truth = blurred_marker(37, 110, (0.0, 0.0), 1.0, seed=3)
    seed = truth + np.array([[0.6, -0.4], [-0.3, 0.5], [0.4, 0.4], [-0.5, -0.2]])
    off = refine.refine(img, seed, 37, CANVAS, origin=origin, border_cells=B)
    for v in ((0.0, 0.0), (1.2, -0.9)):                       # still, and slower than the window's dead zone
        on = refine.refine(img, seed, 37, CANVAS, origin=origin, border_cells=B, velocity=v, exposure_fraction=1.0,
                           sigma_optics=0.5)
        assert np.array_equal(on.corners, off.corners)
        assert (on.edge_rms_px, on.downweighted_frac, on.shift_px, on.n_used) == \
               (off.edge_rms_px, off.downweighted_frac, off.shift_px, off.n_used)


def test_a_velocity_needs_the_exposure_fraction_and_the_optics_sigma():
    img, origin, truth = blurred_marker(37, 110, (0.0, 0.0), 1.0, seed=3)
    with pytest.raises(ValueError, match="exposure_fraction"):
        refine.refine(img, truth, 37, CANVAS, origin=origin, border_cells=B, velocity=(5.0, 0.0), sigma_optics=0.5)
    with pytest.raises(ValueError, match="sigma_optics"):
        refine.refine(img, truth, 37, CANVAS, origin=origin, border_cells=B, velocity=(5.0, 0.0),
                      exposure_fraction=1.0)


# ------------------------------------------------------------------ the window rule and its cap
def test_the_window_widens_with_the_blur_across_each_edge_only():
    n = np.array([[1.0, 0.0], [0.0, 1.0]])                    # an edge across the motion, one along it
    k, w = refine.blur_half_windows(n, 1.5, 15.0, (12.0, 0.0), 1.0, 0.5)
    assert w.tolist() == [12.0, 0.0]
    assert k[1] == 0                                           # the edge parallel to the motion keeps the window
    assert 1.5 + k[0] * refine.PROFILE_STEP >= 1.5 + 12.0 / 2 - 1 - 1e-9     # covers the ramp less the dead zone
    k0, _ = refine.blur_half_windows(n, 1.5, 15.0, (1.9, 0.0), 1.0, 0.5)    # w <= 2 px: the dead zone, no change
    assert (k0 == 0).all()


@pytest.mark.parametrize("half,cell,sig", [(1.0, 4.0, 0.5), (2.0, 8.0, 1.84), (4.0, 15.0, 0.48), (3.2, 11.0, 1.84)])
def test_the_window_never_reaches_a_same_polarity_neighbour(half, cell, sig):
    """Same-polarity edges are >= 2 cells apart. The window plus the half ramp plus 2 sigma of optics stays short of
    the next one wherever the cap is above the production window, and is the production window otherwise."""
    ang = np.linspace(0, np.pi, 13)
    n = np.stack([np.cos(ang), np.sin(ang)], 1)
    for speed in np.linspace(0, 60, 121):
        k, w = refine.blur_half_windows(n, half, cell, (speed, 0.0), 1.0, sig)
        h = half + k * refine.PROFILE_STEP
        assert (h >= half).all()
        cap = 2 * cell - w / 2 - 2 * sig - 0.5
        above = cap > half
        assert (h[above] + w[above] / 2 + 2 * sig < 2 * cell).all()
        assert (h[~above] == half).all()


# ------------------------------------------------------------------ velocity from the detector track
T60 = 1001 / 60000


def _track(pos, sq=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)):
    return np.stack([sq + np.asarray(p, float) for p in pos])


def test_velocity_from_the_nearest_detections_in_time_of_the_same_id():
    from arucotrack.refinement.refine_cache import detector_velocity
    seq = np.array([5, 6, 7, 6, 11, 20, 20, 30, 31])
    ids = np.array([1, 1, 1, 2, 1, 1, 1, 3, 3])
    corners = _track([(0, 0), (3, 1), (8, 3), (50, 50), (20, 20), (0, 0), (1, 1), (0, 0), (4, -2)])
    corners[7, 0, 0] = np.nan                                     # id 3 at seq 30: not finite
    v, kind = detector_velocity(seq, ids, corners, t=seq * T60, period=T60)
    assert np.allclose(v[1], [4.0, 1.5]) and kind[1] == 2         # central: (c[seq 7] - c[seq 5]) / 2 frames
    assert np.allclose(v[0], [3.0, 1.0]) and kind[0] == 1         # first: forward difference
    assert np.allclose(v[2], [5.0, 2.0]) and kind[2] == 1         # seq 11 is 4 frames on: too far, backward only
    assert kind[4] == 0 and np.isnan(v[4]).all()                  # seq 11: nearest is 4 frames back
    assert kind[3] == 0                                           # isolated (another id at seq 6)
    assert kind[5] == 0 and kind[6] == 0                          # a duplicated (seq, id) is never paired
    assert kind[7] == 0 and kind[8] == 0                          # its only neighbour is not finite
    usable = np.ones(len(seq), bool); usable[2] = False           # e.g. a row whose corners are in another image
    v2, kind2 = detector_velocity(seq, ids, corners, usable=usable, t=seq * T60, period=T60)
    assert kind2[2] == 0 and kind2[1] == 1 and np.allclose(v2[1], [3.0, 1.0])


def test_a_dropped_frame_gives_the_same_velocity_as_no_drop():
    """seq is the packet ordinal: after a drop, seq + 1 is two frame intervals later. The velocity comes from the
    timestamps, in px per nominal frame, so a drop changes nothing. A neighbour 2 intervals away is still used, one 3
    intervals away is not (limit 2.5)."""
    from arucotrack.refinement.refine_cache import detector_velocity
    step = np.array([5.0, -2.0])                                  # px per nominal frame
    frames_t = np.array([0, 1, 2, 4, 5, 6])                       # frame 3 dropped: seq 3 is time 4
    seq = np.arange(6)
    corners = _track([k * step for k in frames_t])
    v, kind = detector_velocity(seq, np.full(6, 9), corners, t=0.3 + frames_t * T60, period=T60)
    assert (kind == np.array([1, 2, 2, 2, 2, 1])).all()
    assert np.allclose(v, step[None, :])
    gap3 = np.array([0, 1, 2, 5, 6, 7])                           # two frames lost: 3 intervals
    v3, kind3 = detector_velocity(seq, np.full(6, 9), _track([k * step for k in gap3]), t=gap3 * T60, period=T60)
    assert (kind3 == np.array([1, 2, 1, 1, 2, 1])).all()
    assert np.allclose(v3, step[None, :])


def test_the_nominal_frame_period_survives_millisecond_timestamps_and_drops():
    """Camera PTS can be stored to the millisecond (16 / 17 ms steps) and frames drop. The plain median interval reads
    0.017 s, 1.7 % long. The period is the mean of the single-frame intervals around the median."""
    from arucotrack.refinement.refine_cache import nominal_frame_period
    true = 1 / 59.8
    t = 0.253 + np.arange(1300) * true
    t = np.delete(t, [100, 400, 401, 900])                        # drops
    pts = np.round(t, 3)
    assert abs(np.median(np.diff(pts)) - true) > 2e-4             # the plain median is off
    assert abs(nominal_frame_period(pts) - true) < 2e-5
    ntsc = np.arange(2000) * 1001 / 60000
    assert abs(nominal_frame_period(np.round(ntsc, 6)) - 1001 / 60000) < 1e-7
