"""Test helper: synthetic markers with exact sub-pixel truth.

Corner *accuracy* can only be measured where the answer is known. Real footage
has no annotated corners, and the rigid-square residual, the one annotation-free
measure available, is blind to a uniform scale by construction. A scale error
of the corners is exactly what needs measuring, so it is measured on rendered
markers.

**Rendered through the camera model, distortion included.** Projecting a marker's
edge midpoints and comparing them with the midpoint of its corners shows that the
edges are bowed by 0.12 px at the optical axis, 0.21 px at r = 0.44 and 0.37 px
at r = 0.75 (reference calibration of `conftest.py`). A straight-line contour fit
(what OpenCV does) and a homography fit (what `refine` does) are both wrong by
that much, and the error grows with radius. A homography-only renderer would hide
the effect that is to be measured.

Inverting the distortion is therefore part of the renderer, and
`cv2.undistortPoints` is not accurate enough for it: its fixed iteration count
leaves a radius-dependent residual, the same trap as in `geometry.to_floor`.
`undistort_newton` below iterates to convergence and reports the residual it
achieved.

The appearance defaults follow real captures: the dark level of a marker patch is
44 +/- 13 and the light level 113 +/- 25 (over 400 detections), so the contrast is
only about 63 grey levels. The equivalent edge width of 2.6-3.2 px corresponds to
a Gaussian sigma of width / sqrt(2 pi), about 1.0-1.3 px.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from arucotrack.calibration.camera import Calib
from arucotrack.geometry.projection import undistort_newton
from arucotrack.refinement.refine import CELLS, cell_grid

DARK, LIGHT, CONTRAST_SD = 44.0, 113.0, 8.0
BLUR_SIGMA_PX = 1.15            # equivalent width 2.9 px / sqrt(2 pi)
READ_NOISE = 2.5


@dataclass
class Render:
    patch: np.ndarray            # uint8, (P, P)
    origin: np.ndarray           # (2,) image coords of patch pixel (0,0)
    corners: np.ndarray          # (4,2) TRUE outer corners, full image coords
    side_px: float
    radius: float


def project_outer(calib: Calib, rvec, tvec) -> np.ndarray:
    """The four outer corners of the black square in image px, TL TR BR BL."""
    h = calib.marker_side / 2
    outer = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    proj, _ = cv2.projectPoints(outer, np.asarray(rvec, np.float64).reshape(3, 1),
                                np.asarray(tvec, np.float64).reshape(3, 1), calib.K, calib.D)
    return proj.reshape(4, 2)


def subsample_ideal(calib: Calib, origin: np.ndarray, P: int, S: int) -> np.ndarray:
    """Ideal-plane coordinates of every sub-sample of a P x P patch - shared across the
    time samples of one exposure, which is what makes integrated frames affordable."""
    ii, jj = np.meshgrid(np.arange(P * S), np.arange(P * S), indexing="ij")
    px = origin[0] + (jj.ravel() + 0.5) / S
    py = origin[1] + (ii.ravel() + 0.5) / S
    ideal, _ = undistort_newton(calib.K, calib.D, np.stack([px, py], 1))
    return ideal


def render_raw(calib: Calib, marker_id: int, rvec, tvec, origin: np.ndarray, P: int,
               supersample: int = 4, dark: float = DARK, light: float = LIGHT, bg: float | None = None,
               margin_cells: float = 0.0, margin_level: float | None = None, border_cells: float = 1.0,
               bg_image: np.ndarray | None = None, ideal: np.ndarray | None = None,
               print_warp: np.ndarray | None = None) -> np.ndarray:
    """Area-integrated marker image on a fixed P x P patch at `origin` - no blur, no noise, float.

    The building block for stills (`render`) and for exposures integrated over
    time (`synthvideo`): several of these at different poses are averaged before
    the optical blur and the sensor noise are applied once. `bg_image`, if given,
    is a P x P background the marker is composited onto (sampled at the same
    sub-pixel positions) instead of the flat `bg` level.
    """
    g = cell_grid(marker_id)
    M = calib.marker_side
    h = M / 2
    span = 6.0 + 2.0 * border_cells
    lo_c = 1.0 - border_cells
    S = supersample
    ii, jj = np.meshgrid(np.arange(P * S), np.arange(P * S), indexing="ij")
    if ideal is None:
        ideal = subsample_ideal(calib, origin, P, S)
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    t = np.asarray(tvec, np.float64).reshape(3)
    pitch = M / span
    A = np.array([[pitch, 0, -h - lo_c * pitch], [0, -pitch, h + lo_c * pitch], [0, 0, 1]], np.float64)
    Hm = np.column_stack([R[:, 0], R[:, 1], t])
    Hinv = np.linalg.inv(Hm @ A)
    q = np.concatenate([ideal, np.ones((len(ideal), 1))], 1) @ Hinv.T
    cell = q[:, :2] / q[:, 2:3]
    hi_c = lo_c + span
    in_black = ((cell[:, 0] >= lo_c) & (cell[:, 0] < hi_c) & (cell[:, 1] >= lo_c) & (cell[:, 1] < hi_c))
    # Print imperfection: the interior grid is not exactly where an ideal print puts
    # it. `print_warp` = 12 coefficients of a quadratic field (dx, dy) in cell units
    # over centred coordinates u, v in [-1, 1]: [1, u, v, u^2, uv, v^2] x 2. The
    # black square's outer edge is left exact, so the true corners stay well defined.
    cd = cell
    if print_warp is not None:
        u = (cell[:, 0] - CELLS / 2) / (CELLS / 2)
        v = (cell[:, 1] - CELLS / 2) / (CELLS / 2)
        B = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
        w = np.asarray(print_warp, np.float64).reshape(2, 6)
        cd = cell + np.stack([B @ w[0], B @ w[1]], 1)
    in_data = ((cd[:, 0] >= 1.0) & (cd[:, 0] < CELLS - 1) & (cd[:, 1] >= 1.0) & (cd[:, 1] < CELLS - 1))
    ci = np.clip(cd[:, 1].astype(int), 0, CELLS - 1)
    cj = np.clip(cd[:, 0].astype(int), 0, CELLS - 1)
    val = np.where(in_data & (g[ci, cj] > 0), light, dark)
    if bg_image is not None:
        # nearest background pixel for every sub-sample (the background is already
        # a real, blurred image, resampling it finer adds nothing)
        bi = np.clip((ii.ravel() // S), 0, P - 1)
        bj = np.clip((jj.ravel() // S), 0, P - 1)
        background = np.asarray(bg_image, np.float64)[bi, bj]
    else:
        background = light if bg is None else bg
    if margin_cells > 0:
        lvl = light if margin_level is None else margin_level
        in_margin = ((cell[:, 0] >= lo_c - margin_cells) & (cell[:, 0] < hi_c + margin_cells) &
                     (cell[:, 1] >= lo_c - margin_cells) & (cell[:, 1] < hi_c + margin_cells))
        val = np.where(in_black, val, np.where(in_margin, lvl, background))
    else:
        val = np.where(in_black, val, background)
    fine = val.reshape(P * S, P * S)
    return fine.reshape(P, S, P, S).mean(axis=(1, 3))


def render(calib: Calib, marker_id: int, rvec: np.ndarray, tvec: np.ndarray,
           blur_sigma: float = BLUR_SIGMA_PX, noise_sd: float = READ_NOISE,
           supersample: int = 4, pad: float = 14.0, rng=None,
           dark: float = DARK, light: float = LIGHT, bg: float | None = None,
           margin_cells: float = 0.0, margin_level: float | None = None,
           border_cells: float = 1.0) -> Render | None:
    """One marker at an exact pose, area-integrated then blurred and noised.

    Rendering is done by inverse mapping: every sub-pixel sample is taken back
    through the *full* distortion model to the marker plane, so the cell edges
    land exactly where the camera model says they do, bow and all.
    """
    rng = rng or np.random.default_rng(0)
    corners = project_outer(calib, rvec, tvec)
    if not np.isfinite(corners).all():
        return None
    lo = corners.min(0) - pad
    hi = corners.max(0) + pad
    P = int(np.ceil(max(hi[0] - lo[0], hi[1] - lo[1])))
    if P < 12 or P > 600:
        return None
    origin = np.floor(lo)
    img = render_raw(calib, marker_id, rvec, tvec, origin, P, supersample, dark, light, bg,
                     margin_cells, margin_level, border_cells)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_sd > 0:
        img = img + rng.normal(0, noise_sd, img.shape)
    patch = np.clip(img, 0, 255).astype(np.uint8)
    side = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    cen = corners.mean(0)
    rad = float(np.hypot(cen[0] - calib.cx, cen[1] - calib.cy) / calib.half_diag)
    return Render(patch, origin, corners, side, rad)


def _orient(d: np.ndarray, tilt_deg: float, yaw: float) -> np.ndarray:
    """rvec with the marker normal facing back along viewing ray `d`, then tilted and yawed."""
    z = -d
    a = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s, c = np.linalg.norm(v), float(np.dot(a, z))
    if s < 1e-9:
        R0 = np.eye(3)
    else:
        Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R0 = np.eye(3) + Kx + Kx @ Kx * ((1 - c) / (s * s))
    tilt_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    Rt = cv2.Rodrigues(tilt_axis * np.radians(tilt_deg))[0]
    Ry = cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]
    return cv2.Rodrigues(R0 @ Rt @ Ry)[0].ravel()


def pose_at(calib: Calib, radius: float, side_px: float, tilt_deg: float, yaw: float,
            rng=None, azimuth: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A pose that puts the marker at image radius `radius` (0..1 of the half
    diagonal) with the requested *achieved* apparent side.

    The paraxial f*M/d is only a first guess: off-axis the local magnification is
    not f, and using it directly can miss the requested side by a factor of three
    (342 px achieved for 107 asked). Secant-iterate on the achieved side instead (max rel err 1e-3).
    """
    rng = rng or np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi) if azimuth is None else azimuth
    uv = np.array([[calib.cx + radius * calib.half_diag * np.cos(ang),
                    calib.cy + radius * calib.half_diag * np.sin(ang)]])
    ideal, _ = undistort_newton(calib.K, calib.D, uv)
    d = np.array([ideal[0, 0], ideal[0, 1], 1.0])
    d /= np.linalg.norm(d)
    h = calib.marker_side / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    dist = calib.K[0, 0] * calib.marker_side / side_px
    rv = _orient(d, tilt_deg, yaw)
    for _ in range(8):
        pr, _ = cv2.projectPoints(obj, rv, d * dist, calib.K, calib.D)
        c = pr.reshape(4, 2)
        got = float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))
        if not np.isfinite(got) or got < 1e-6:
            break
        if abs(got - side_px) / side_px < 1e-3:
            break
        dist *= got / side_px
    return rv, d * dist
