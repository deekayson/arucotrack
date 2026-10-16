"""Interior-edge corner refinement and the synthetic renderer it is measured on.

A refiner can look plausible and still be wrong, and only ground truth shows it.
The corners are therefore checked on rendered markers (`tests/synthetic.py`)
whose pose, and so whose true corners, are exact.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from arucotrack.geometry import projection
from arucotrack.refinement import refine
from tests import synthetic
from tests.conftest import reference_calib


@pytest.fixture(scope="module")
def calib():
    return reference_calib()


def _pose(calib, radius, side_px, tilt_deg=8.0, yaw=0.4):
    ang = 0.7
    uv = np.array([[calib.cx + radius * calib.half_diag * np.cos(ang),
                    calib.cy + radius * calib.half_diag * np.sin(ang)]])
    ideal, _ = projection.undistort_newton(calib.K, calib.D, uv)
    d = np.array([ideal[0, 0], ideal[0, 1], 1.0])
    d /= np.linalg.norm(d)
    tvec = d * (calib.K[0, 0] * calib.marker_m / side_px)
    z = -d
    a = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s, c = np.linalg.norm(v), float(np.dot(a, z))
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R0 = np.eye(3) if s < 1e-9 else np.eye(3) + Vx + Vx @ Vx * ((1 - c) / (s * s))
    R = R0 @ cv2.Rodrigues(np.array([np.cos(yaw), np.sin(yaw), 0.0]) * np.radians(tilt_deg))[0]
    return cv2.Rodrigues(R)[0].ravel(), tvec


# ---------------------------------------------------------------- the dictionary

def test_polarity_is_exactly_balanced_for_every_roster_id():
    """The whole mechanism rests on this, and it is structural, not luck.

    The one-cell black border forces the first and last cell of every interior
    scan line to be black, so each line carries equal numbers of black-to-white
    and white-to-black transitions and a constant edge bias cancels exactly.
    """
    for mid in [17, 18, 19, 20, 22, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 43, 44]:
        g = refine.cell_grid(mid)
        assert g[0].sum() == 0 and g[-1].sum() == 0        # border rows black
        assert g[:, 0].sum() == 0 and g[:, -1].sum() == 0  # border columns black
        _, _, pol = refine.interior_samples(g, per_edge=1)
        assert pol.sum() == 0, f"id {mid} has a polarity imbalance"
        assert len(pol) >= 30, f"id {mid} has only {len(pol)} interior edges"


# ---------------------------------------------------------------- the renderer

def test_undistort_newton_round_trips_and_handles_the_rational_model(calib):
    assert np.count_nonzero(np.ravel(calib.D)[5:8]) > 0, "expected the rational model"
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 2000)
    rr = np.sqrt(rng.uniform(0, 1, 2000)) * 2.4
    ideal = np.stack([rr * np.cos(th), rr * np.sin(th)], 1)
    pix = projection.distort_points(calib.K, calib.D, ideal)
    m = ((pix[:, 0] >= 0) & (pix[:, 0] < 2592) & (pix[:, 1] >= 0) & (pix[:, 1] < 1944))
    back, resid = projection.undistort_newton(calib.K, calib.D, pix[m])
    err = np.linalg.norm(back - ideal[m], axis=1) * calib.K[0, 0]
    assert err.max() < 1e-6, f"worst round-trip {err.max():.2e} px"
    assert resid < 1e-12


def test_renderer_pixel_convention_matches_the_detector(calib):
    """A rendered edge must land where ArUco says it does, or every corner
    number is off by a flat half pixel - which a homography hides as a
    translation, so it will not show up as a bad fit."""
    rvec, tvec = _pose(calib, 0.25, 95.0, tilt_deg=6.0)
    r = synthetic.render(calib, 37, rvec, tvec, noise_sd=0.0, rng=np.random.default_rng(0))
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    cs, ids, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(r.patch)
    assert ids is not None and 37 in ids.ravel()
    got = cs[list(ids.ravel()).index(37)].reshape(4, 2).astype(np.float64)
    truth = r.corners - r.origin
    best = min(np.linalg.norm(np.roll(got, k, axis=0) - truth, axis=1).mean() for k in range(4))
    assert best < 0.25, f"detector disagrees with rendered truth by {best:.3f} px"


def test_rendered_marker_is_detectable_across_the_real_size_range(calib):
    rng = np.random.default_rng(2)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    # `_pose` sizes by the paraxial f*M/d, so the achieved side differs off-axis
    # where the local magnification is not f. The tests use the
    # achieved `side_px`, never the requested one.
    achieved = []
    for side in (40.0, 70.0, 100.0):
        rvec, tvec = _pose(calib, 0.3, side)
        r = synthetic.render(calib, 37, rvec, tvec, rng=rng)
        assert r is not None
        achieved.append(r.side_px)
        cs, ids, _ = det.detectMarkers(r.patch)
        assert ids is not None and 37 in ids.ravel(), f"not detected at {r.side_px:.0f} px"
    assert achieved == sorted(achieved)
    assert min(achieved) < 70 < max(achieved), achieved


# ---------------------------------------------------------------- the refiner

@pytest.mark.parametrize("radius,side", [(0.2, 90.0), (0.45, 70.0), (0.7, 60.0)])
def test_refiner_initialised_at_truth_stays_at_truth(calib, radius, side):
    """The basic check, which fails on an inverted edge polarity, on a homography
    fitted in pixels where the edges are bowed, and on a half-pixel sampling offset."""
    rvec, tvec = _pose(calib, radius, side)
    r = synthetic.render(calib, 37, rvec, tvec, rng=np.random.default_rng(3))
    truth = r.corners - r.origin
    ref = refine.refine(r.patch, truth, 37, calib, origin=r.origin, border_cells=1.0)
    assert ref.converged, ref.reason
    assert ref.n_used >= 60
    err = float(np.sqrt(((ref.corners - truth) ** 2).sum(1).mean()))
    assert err < 0.45, f"refiner walked {err:.3f} px away from truth"


def test_refiner_is_not_merely_following_its_initialisation(calib):
    """Perturb the seed inward. A corrector returns toward truth, a smoother
    follows the seed. Without this, a refinement that just reproduces the
    biased quad would look like a success."""
    rvec, tvec = _pose(calib, 0.3, 90.0)
    r = synthetic.render(calib, 37, rvec, tvec, rng=np.random.default_rng(4))
    truth = r.corners - r.origin
    cen = truth.mean(0)
    seed = cen + (truth - cen) * 0.985          # ~0.7 px inward shrink
    seed_err = float(np.sqrt(((seed - truth) ** 2).sum(1).mean()))
    ref = refine.refine(r.patch, seed, 37, calib, origin=r.origin, border_cells=1.0)
    assert ref.converged, ref.reason
    err = float(np.sqrt(((ref.corners - truth) ** 2).sum(1).mean()))
    assert err < seed_err * 0.6, f"seed {seed_err:.3f} px -> refined {err:.3f} px"


def test_border_width_is_load_bearing_and_the_constant_is_applied(calib):
    """Assuming a one-cell border when it is 0.869 is a 3.2 % scale error, worth
    several centimetres of height. Render the measured border and check the default
    constant recovers truth, while the nominal 1.0 does not."""
    rvec, tvec = _pose(calib, 0.3, 95.0)
    r = synthetic.render(calib, 37, rvec, tvec, rng=np.random.default_rng(6),
                     border_cells=refine.BORDER_CELLS)
    truth = r.corners - r.origin
    good = refine.refine(r.patch, truth, 37, calib, origin=r.origin)     # default 0.869
    bad = refine.refine(r.patch, truth, 37, calib, origin=r.origin, border_cells=1.0)
    assert good.converged and bad.converged
    e_good = float(np.sqrt(((good.corners - truth) ** 2).sum(1).mean()))
    e_bad = float(np.sqrt(((bad.corners - truth) ** 2).sum(1).mean()))
    assert e_good < 0.45, f"measured border should recover truth, got {e_good:.3f} px"
    assert e_bad > 3 * e_good, f"nominal border should be clearly wrong: {e_bad:.3f} px"
    side_true = np.mean([np.linalg.norm(truth[j] - truth[(j + 1) % 4]) for j in range(4)])
    side_bad = np.mean([np.linalg.norm(bad.corners[j] - bad.corners[(j + 1) % 4]) for j in range(4)])
    assert (side_bad - side_true) / side_true > 0.02, "nominal border over-expands the quad"


def test_refiner_reports_failure_rather_than_guessing(calib):
    rvec, tvec = _pose(calib, 0.3, 90.0)
    r = synthetic.render(calib, 37, rvec, tvec, rng=np.random.default_rng(5))
    flat = np.full_like(r.patch, 100)
    ref = refine.refine(flat, r.corners - r.origin, 37, calib, origin=r.origin)
    assert not ref.converged
    assert ref.reason


# ---------------------------------------------------------------- edge_samples

def test_edge_samples_agree_with_edge_residuals(calib):
    rvec, tvec = _pose(calib, 0.3, 70.0)
    r = synthetic.render(calib, 40, rvec, tvec, noise_sd=0.0, rng=np.random.default_rng(0))
    corners = r.corners - r.origin
    d0, ok0 = refine.edge_residuals(r.patch, corners, 40, calib, origin=r.origin)
    s = refine.edge_samples(r.patch, corners, 40, calib, origin=r.origin)
    np.testing.assert_array_equal(s.ok, ok0)
    np.testing.assert_allclose(s.d[s.ok], d0[ok0])
    assert s.pos.shape == (len(d0), 2) and s.normal.shape == (len(d0), 2)
    np.testing.assert_allclose(np.linalg.norm(s.normal, axis=1), 1.0, atol=1e-9)


def test_edge_samples_lie_inside_the_quad(calib):
    rvec, tvec = _pose(calib, 0.0, 80.0)
    r = synthetic.render(calib, 18, rvec, tvec, noise_sd=0.0, rng=np.random.default_rng(0))
    corners = r.corners - r.origin
    s = refine.edge_samples(r.patch, corners, 18, calib, origin=r.origin)
    path = cv2.convexHull(corners.astype(np.float32))
    assert len(s.pos) > 0
    assert all(cv2.pointPolygonTest(path, tuple(map(float, p)), False) > 0 for p in s.pos)


def test_marker_image_with_a_one_cell_border_is_the_dictionary_marker():
    from tests.conftest import marker_image
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    assert np.array_equal(marker_image(37, 160, 1.0), cv2.aruco.generateImageMarker(dic, 37, 160))
