"""Sub-pixel corner refinement from the marker's interior edges (the paper, Section 3.4, "Corner refinement").

The detector's corner estimate comes from the marker's outer boundary, one isolated dark-to-light edge.
Mallon & Whelan (2007) show why that is a poor choice: a checkerboard X-junction is free of edge-localisation bias
*because* it sits between four alternating fields, so the bias of each quadrant cancels against its neighbours. An
isolated boundary gets no such cancellation. Its bias usually shrinks the quad uniformly, which moves the marker away
from the camera by a constant fraction and so reads as a height error.

The interior gets most of the cancellation back. A marker of a 6x6 dictionary such as `DICT_ARUCO_MIP_36h12`
is a 6x6 data grid inside a one-cell black border, so 8x8 cells. The border makes the first and last cell of every
interior grid line black, so each line has equal numbers of dark-to-light and light-to-dark edges, for every id. A
blur or threshold bias that shifts every edge towards the dark side therefore largely cancels over the interior. The
edges against the black border do not cancel, and the scale they leave is absorbed by the border constant b
(`BORDER_CELLS`). A 21-id set of this dictionary has 34-50 interior edges per marker (42.7 +/- 4.1).

True X-junctions are not a usable alternative: the same 21-id set has only 0-8 per marker (mean 2.9), some ids none
at all. The count also varies with the id, so a refinement built on them would be conditioned differently for each
marker and would add a per-marker spread of its own. The edge count varies far less.

Corner refinement of fiducial markers usually works on the outer edge only (for example Kallwies et al., ICRA 2020),
where a blur or threshold bias shifts all four sides alike. This module applies the Mallon & Whelan cancellation
argument to the interior edges of the marker instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CELLS = 8          # 6x6 data bits plus a one-cell black border
BORDER = 1


def cell_grid(marker_id: int, dictionary=None) -> np.ndarray:
    """The marker's 8x8 cell pattern, 0 black and 1 white, border included."""
    dictionary = dictionary or cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
    img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), CELLS, borderBits=BORDER)
    return (img // 255).astype(np.uint8)


def interior_samples(g: np.ndarray, per_edge: int = 3, inset: float = 0.22):
    """Sample points on every interior colour-change edge, in cell coordinates.

    Returns `(pts, tang, pol)`: positions in cell units (origin at the marker's
    outer corner, `CELLS` across), the unit tangent of the edge each point sits
    on, and the polarity: `+1` when the cell on the left of the tangent is
    black, so the intensity rises across the edge in the direction of the normal.

    Points are inset from the edge ends so that a sample never sits on the
    junction with a perpendicular edge, where the profile is not a step.
    """
    pts, tang, pol = [], [], []
    frac = np.linspace(inset, 1.0 - inset, per_edge)
    for i in range(CELLS):                      # vertical edges: between (i,j) and (i,j+1)
        for j in range(CELLS - 1):
            if g[i, j] == g[i, j + 1]:
                continue
            x = float(j + 1)
            for f in frac:
                pts.append((x, i + f)); tang.append((0.0, 1.0))
                # n = (-t_y, t_x) points toward the LEFT cell (i, j). Intensity
                # rises along +n when that cell is the white one
                pol.append(1 if g[i, j] == 1 else -1)
    for i in range(CELLS - 1):                  # horizontal edges: between (i,j) and (i+1,j)
        for j in range(CELLS):
            if g[i, j] == g[i + 1, j]:
                continue
            y = float(i + 1)
            for f in frac:
                pts.append((j + f, y)); tang.append((1.0, 0.0))
                # n points toward the LOWER cell (i+1, j)
                pol.append(1 if g[i + 1, j] == 1 else -1)
    return (np.asarray(pts, np.float64), np.asarray(tang, np.float64),
            np.asarray(pol, np.float64))


def _bilinear(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample at a continuous image coordinate, NaN outside.

    Pixel `j` spans `[j, j+1)` and is therefore centred at `j + 0.5`, which is
    the convention the ArUco detectors were measured to use (a synthetic edge at
    patch x = 14.605 renders pixel 14 exactly 3/8 dark, and `CORNER_REFINE_
    APRILTAG` recovers the corner to 0.057 px with no offset). Interpolating as
    if the centres sat on the integers costs a flat half pixel per axis, which
    is 0.71 px on a corner, and a homography absorbs it as a pure translation,
    so it hides as a perfect fit.
    """
    h, w = img.shape
    x, y = xy[:, 0] - 0.5, xy[:, 1] - 0.5
    ok = (x >= 0) & (x <= w - 2) & (y >= 0) & (y <= h - 2)
    out = np.full(len(xy), np.nan)
    if not ok.any():
        return out
    xs, ys = x[ok], y[ok]
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = xs - x0; fy = ys - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
         img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    out[ok] = v
    return out


def _apply(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.concatenate([p, np.ones((len(p), 1))], 1) @ H.T
    return q[:, :2] / q[:, 2:3]


def _predict_edges(H: np.ndarray, pts: np.ndarray, tang: np.ndarray, pol: np.ndarray, K, D, eps: float = 1e-4):
    """Where the homography puts each edge sample: `(q_id, q_img, n_img)`, the ideal-plane position, image
    position, and the unit image normal oriented by polarity (intensity rises along +n_img). Shared by
    the fit and the quality pass, so the two cannot drift apart."""
    from ..geometry import projection
    q_id = _apply(H, pts)
    t_id = _apply(H, pts + tang * eps) - q_id
    q_img = projection.distort_points(K, D, q_id)
    t_img = projection.distort_points(K, D, q_id + t_id) - q_img
    n_img = np.stack([-t_img[:, 1], t_img[:, 0]], 1)
    ln = np.linalg.norm(n_img, axis=1, keepdims=True)
    return q_id, q_img, n_img / np.where(ln < 1e-12, 1.0, ln) * pol[:, None]


def _half_window(corners: np.ndarray) -> float:
    """Half-width (px) of the edge profiles for a quad: 0.42 of a cell, clipped to [1, 4] px."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    side_px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    return float(np.clip(0.42 * side_px / CELLS, 1.0, 4.0))


def _fit_homography_to_lines(H: np.ndarray, p: np.ndarray, meas: np.ndarray, nrm: np.ndarray,
                             cauchy_c: float = 1.5, steps: int = 6):
    """Gauss-Newton on the 8 free homography parameters, point-to-line.

    An edge sample localises its edge only *across* the edge. Where it sits
    along the edge is unknown. So the residual is the normal component alone,
    `n . (H p - m)`, and the tangential direction carries no constraint. Fitting
    point-to-point instead would invent one, and with three samples per edge
    that invented constraint would quietly dominate.

    Cauchy-weighted IRLS, so a few bad profiles cannot drag it.
    """
    h = np.asarray(H, np.float64).ravel().copy()
    h = h / h[8]
    P = np.concatenate([p, np.ones((len(p), 1))], 1)
    rms = float("nan")
    for _ in range(steps):
        s = P @ h[6:9]
        if np.any(np.abs(s) < 1e-9):
            return None, float("nan")
        qx = (P @ h[0:3]) / s
        qy = (P @ h[3:6]) / s
        r = nrm[:, 0] * (qx - meas[:, 0]) + nrm[:, 1] * (qy - meas[:, 1])
        sd = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = 1.0 / (1.0 + (r / (cauchy_c * sd)) ** 2)
        J = np.empty((len(p), 8))
        J[:, 0:3] = (nrm[:, 0] / s)[:, None] * P
        J[:, 3:6] = (nrm[:, 1] / s)[:, None] * P
        J[:, 6:8] = (-(nrm[:, 0] * qx + nrm[:, 1] * qy) / s)[:, None] * P[:, :2]
        sw = np.sqrt(w)
        dh, *_ = np.linalg.lstsq(J * sw[:, None], -r * sw, rcond=None)
        h[:8] += dh
        rms = float(np.sqrt(np.average(r ** 2, weights=w)))
        if np.max(np.abs(dh)) < 1e-10:
            break
    return h.reshape(3, 3), rms


FAR_PX = 1.0
"""An interior edge sample further than this from its fitted line counts as rejected in
`Refinement.downweighted_frac`. A healthy fit's edge residual is 0.2-0.4 px rms (measured on an undistorted
view of a fisheye camera and on a global-shutter camera), so 1 px is about 3-4 sigma, where the Cauchy IRLS weight is already
below about 0.15."""


@dataclass
class Refinement:
    corners: np.ndarray          # (4,2) refined outer corners
    n_used: int                  # edge samples that survived
    n_total: int
    rms_px: float                # weighted IRLS residual in the IDEAL plane, units 1/f, not px (see refine)
    shift_px: float              # how far the corners moved (mean over the four corners, image px)
    converged: bool
    reason: str = ""
    edge_rms_px: float = float("nan")         # unweighted rms edge-to-fitted-line distance, image px
    downweighted_frac: float = float("nan")   # interior samples unusable or > FAR_PX from their line


@dataclass(frozen=True)
class EdgeSamples:
    pos: np.ndarray      # (n, 2) predicted sample position on its fitted line, px of `gray`
    normal: np.ndarray   # (n, 2) unit image normal, intensity rises along +normal
    d: np.ndarray        # (n,) signed offset of the measured edge along `normal`, px, NaN where not ok
    ok: np.ndarray       # (n,) bool


def edge_samples(gray: np.ndarray, corners: np.ndarray, marker_id: int, calib,
                 origin: np.ndarray | tuple = (0.0, 0.0), dictionary=None, per_edge: int = 3,
                 border_cells: float | None = None) -> EdgeSamples:
    """`edge_residuals` with the sample geometry: where each interior edge sample is predicted, its normal, and the
    measured offset, for plotting a fit. `edge_residuals` is this, reduced to `(d, ok)`, so the two cannot drift."""
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    origin = np.asarray(origin, np.float64).reshape(2)
    g = cell_grid(marker_id, dictionary)
    pts, tang, pol = interior_samples(g, per_edge=per_edge)
    H, _ = _H_from_corners(corners + origin, calib, BORDER_CELLS if border_cells is None else border_cells)
    _, q_img, n_img = _predict_edges(H, pts, tang, pol, calib.K, calib.D)
    d, ok = _edge_offsets(gray, q_img - origin, n_img, _half_window(corners))
    return EdgeSamples(q_img - origin, n_img, d, ok)


def edge_residuals(gray: np.ndarray, corners: np.ndarray, marker_id: int, calib,
                   origin: np.ndarray | tuple = (0.0, 0.0), dictionary=None, per_edge: int = 3,
                   border_cells: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance, in px of `gray`, of every interior edge sample from where the quad puts it.

    The quad's homography (outer corners at `border_cells`) predicts each interior cell-boundary line.
    Each edge profile is re-sampled across that line along a unit image normal, and its gradient
    centroid is the edge's offset from the line. Returns `(d, ok)` over all interior samples (`d` NaN
    where `ok` is False: no edge of sufficient contrast, or out of the crop). This is the quality
    measure of a refinement (the same estimator the fit uses, evaluated once at the result,
    unweighted), and it applies equally to the detector's quad, which is how the two are compared.

    The measure saturates. The profile spans +/- the half-window (<= 4 px, `_half_window` of the quad passed
    in). For a refinement that is the refined quad, while the fit sized it from the seed. The rule is the
    same, so the windows differ only by the corner shift. An edge further off than the half-window is not
    measured at all: it comes back `ok=False` and counts only in `downweighted_frac`, never in
    `edge_rms_px`. A gross misfit can therefore show a modest edge rms over the edges it still finds,
    which is why the gate needs both criteria (`refine_cache.MAX_DOWNWEIGHTED_FRAC`).
    """
    s = edge_samples(gray, corners, marker_id, calib, origin=origin, dictionary=dictionary,
                     per_edge=per_edge, border_cells=border_cells)
    return s.d, s.ok


def edge_quality(d: np.ndarray, ok: np.ndarray, far_px: float = FAR_PX) -> tuple[float, float]:
    """`(edge_rms_px, downweighted_frac)` from `edge_residuals`."""
    n = len(d)
    if not n or not ok.any():
        return float("nan"), 1.0
    dd = d[ok]
    return float(np.sqrt(np.mean(dd ** 2))), float(((~ok).sum() + (np.abs(dd) > far_px).sum()) / n)


def _edge_offsets(gray: np.ndarray, q_img: np.ndarray, n_img: np.ndarray,
                  half_px: float, step: float = 0.25, min_contrast: float = 12.0):
    """Signed sub-pixel offset of each edge from where it is predicted to be.

    The estimator is the **gradient centroid** across the profile, unbiased for
    any symmetric blur. Whatever asymmetric part remains cancels between the
    exactly equal numbers of opposite-polarity edges.
    """
    s = np.arange(-half_px, half_px + 1e-9, step)
    prof = np.stack([_bilinear(gray, q_img + n_img * si) for si in s], 1)
    good = np.isfinite(prof).all(1)
    grad = np.gradient(prof, step, axis=1)
    grad = np.where(grad > 0, grad, 0.0)
    wsum = grad.sum(1)
    contrast = np.nanmax(prof, axis=1) - np.nanmin(prof, axis=1)
    ok = good & (wsum > 1e-6) & (contrast >= min_contrast)
    d = np.full(len(q_img), np.nan)
    d[ok] = (grad[ok] * s[None, :]).sum(1) / wsum[ok]
    return d, ok


PROFILE_STEP = 0.25
"""Profile sample spacing (px) of `_edge_offsets`. The blur window grows in whole steps of it."""


def blur_half_windows(n_img: np.ndarray, half_px: float, cell_px: float, velocity, exposure_fraction: float,
                      sigma_optics: float, step: float = PROFILE_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Per edge sample, how far the profile window grows for exposure motion blur: `(k, w)`.

    `w = exposure_fraction x |velocity . n|` is the box the edge is smeared over during the exposure, in px
    (`velocity` in px/frame of the image the refiner works on, `n` the sample's unit image normal). The half-window
    becomes `h = half_px + max(0, w/2 - 1)`. The default window `half_px` keeps covering the optics, the extra reaches
    the ramp's ends, and a blur of up to 2 px changes nothing. `h` is capped at
    `max(half_px, 2 cell - w/2 - 2 sigma - 0.5)`: the next same-polarity transition is at least two cells away and its
    ramp reaches `w/2 + 2 sigma` towards this one, so the window never takes in its gradient. Opposite-polarity
    neighbours need no cap, because the clipped (polarity-oriented) gradient excludes them. `h` is rounded UP from
    `half_px` to whole profile steps (`k` steps), so `k == 0` is the default window exactly."""
    n = np.asarray(n_img, np.float64).reshape(-1, 2)
    w = float(exposure_fraction) * np.abs(n @ np.asarray(velocity, np.float64).reshape(2))
    h = half_px + np.maximum(0.0, w / 2 - 1.0)
    cap = np.maximum(half_px, 2 * cell_px - w / 2 - 2 * float(sigma_optics) - 0.5)
    h = np.minimum(h, cap)
    k = np.ceil(np.maximum(h - half_px, 0.0) / step - 1e-9).astype(int)
    return k, w


def _edge_offsets_windowed(gray: np.ndarray, q_img: np.ndarray, n_img: np.ndarray, half_px: float, k: np.ndarray,
                           step: float = PROFILE_STEP, min_contrast: float = 12.0):
    """`_edge_offsets` with a per-sample half-window `half_px + k_i x step`: the same gradient centroid (positive part
    of the polarity-oriented gradient) over each sample's own window. All samples share one grid
    `t_j = -(half_px + K step) + j step` (K = max k). Sample i uses the `|t| <= half_px + k_i step` part of it, with
    one-sided differences at its own window ends, as `np.gradient` takes them on the default grid."""
    k = np.asarray(k, int)
    K = int(k.max()) if len(k) else 0
    t = np.arange(-(half_px + K * step), half_px + K * step + 1e-9, step)
    prof = np.stack([_bilinear(gray, q_img + n_img * ti) for ti in t], 1)
    lo = K - k
    hi = len(t) - 1 - (K - k)
    j = np.arange(len(t))[None, :]
    mask = (j >= lo[:, None]) & (j <= hi[:, None])
    good = np.where(mask, np.isfinite(prof), True).all(1)
    p = np.where(mask & np.isfinite(prof), prof, 0.0)
    grad = np.gradient(p, step, axis=1)
    r = np.arange(len(q_img))
    grad[r, lo] = (p[r, np.minimum(lo + 1, len(t) - 1)] - p[r, lo]) / step
    grad[r, hi] = (p[r, hi] - p[r, np.maximum(hi - 1, 0)]) / step
    grad = np.where(mask & (grad > 0), grad, 0.0)
    wsum = grad.sum(1)
    pm = np.where(mask, prof, np.nan)
    with np.errstate(all="ignore"):
        contrast = np.nanmax(pm, 1) - np.nanmin(pm, 1)
    ok = good & (wsum > 1e-6) & (contrast >= min_contrast)
    d = np.full(len(q_img), np.nan)
    d[ok] = (grad[ok] * t[None, :]).sum(1) / wsum[ok]
    return d, ok


BORDER_CELLS = 0.869
"""Default border constant b: the printed border width, in data-cell widths, for an unmeasured print.

The refiner fits the interior grid (cell boundaries 1..7) and places the outer corners of the black square b cells
further out, so the side of the square is M = (n + 2b) c, with n = 6 data cells of width c. Nominally b = 1, but the
print and the edge response of the optics make the fitted grid end slightly off the visible edge. 0.869 is a default
for an unmeasured print that lies between the values set for the two cameras of 25-Representation (0.855 and 0.892,
the paper, Appendix A). Positions scale with it (b = 1 instead of 0.869 makes the square 3.4 % larger,
so markers read closer to the camera). Set your own value as described in docs/CALIBRATION.md, "Border constant".
"""


def refine(gray: np.ndarray, corners: np.ndarray, marker_id: int, calib,
           origin: np.ndarray | tuple = (0.0, 0.0), dictionary=None,
           iters: int = 3, per_edge: int = 3, cauchy_c: float = 1.5,
           border_cells: float = BORDER_CELLS, velocity=None, exposure_fraction: float | None = None,
           sigma_optics: float | None = None) -> Refinement:
    """Refine one detection's four corners using the marker's interior edges.

    The fit is done in the ideal (undistorted) plane, and that is essential.
    A marker's image is a homography of its cell grid only after the
    distortion is removed. In pixels the cell edges are genuinely bowed (measured
    on one calibrated camera: 0.12 px on axis, 0.37 px at r = 0.75). Fitting a homography
    directly in pixels and extrapolating it out to the outer corners was measured to
    cost 0.7-1.3 px, larger than the bias being corrected. So intensities are sampled
    in the image, where the pixels are, and the geometry is solved in the ideal
    plane, where the model is exact.

    The four outer corners are *output*, not fitted: the homography is fitted to
    the interior and the corners follow from it. Eight numbers in, eight out, so
    this drops into the pipeline ahead of IPPE unchanged.

    Quality: `Refinement.rms_px` is not a pixel quantity: it is the Cauchy-IRLS-weighted rms of the
    last Gauss-Newton step's point-to-line residuals in the ideal (normalised) plane, i.e. in units of 1/f
    (typically 0.0002-0.0004, about 0.2-0.3 px x f). Its weights come from the MAD of the same residuals,
    so it cannot single out a bad fit. The quality measure is `edge_rms_px` (unweighted rms distance of
    the interior edges from the fitted lines, px of `gray`, one evaluation pass at the result) with
    `downweighted_frac` (samples unusable or more than `FAR_PX` from their line). Healthy fits have
    0.2-0.5 px. A seed off by a cell, or motion blur, gives about 1 px.

    Exposure motion blur (opt-in): `velocity` (px/frame on `gray`'s pixel grid, 2-vector) with
    `exposure_fraction` (exposure time / frame interval) and `sigma_optics` (the camera's at-rest optics blur, px)
    widens each edge sample's profile window in the fit only (`blur_half_windows`): an edge smeared over a ramp
    wider than the window is otherwise located wherever the seed puts the window inside the ramp. The quality
    pass keeps the default window, so the gate does not depend on the blur handling. `velocity` None gives the
    refiner without blur handling exactly, and so does any velocity whose blur stays inside the window's dead
    zone (every `k == 0`: the default `_edge_offsets` is called).
    """
    from ..geometry import projection

    if velocity is not None:
        if exposure_fraction is None:
            raise ValueError("refine: a velocity needs the exposure_fraction (exposure time / frame interval)")
        if sigma_optics is None:
            raise ValueError("refine: a velocity needs sigma_optics (the camera's at-rest optics blur, px)")
        velocity = np.asarray(velocity, np.float64).reshape(2)
        if not (np.isfinite(velocity).all() and np.isfinite(exposure_fraction) and exposure_fraction >= 0
                and np.isfinite(sigma_optics) and sigma_optics >= 0):
            raise ValueError(f"refine: velocity {velocity}, exposure_fraction {exposure_fraction} and sigma_optics "
                             f"{sigma_optics} must be finite and non-negative")
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    origin = np.asarray(origin, np.float64).reshape(2)
    g = cell_grid(marker_id, dictionary)
    pts, tang, pol = interior_samples(g, per_edge=per_edge)
    if not len(pts):
        return Refinement(corners, 0, 0, float("nan"), 0.0, False, "no interior edges")

    K, D = calib.K, calib.D
    # the detected quad is the outer edge of the BLACK SQUARE, which sits at
    # cell (1 - border) and (CELLS - 1 + border), not at 0 and CELLS
    lo = 1.0 - border_cells
    hi = CELLS - 1.0 + border_cells
    quad = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], np.float64)
    ideal_c, _ = projection.undistort_newton(K, D, corners + origin)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), ideal_c.astype(np.float32))

    side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    cell_px = side_px / CELLS
    half = _half_window(corners)

    rms, used = float("nan"), 0
    for _ in range(iters):
        q_id, q_img, n_img = _predict_edges(H, pts, tang, pol, K, D)

        kw = (None if velocity is None else
              blur_half_windows(n_img, half, cell_px, velocity, exposure_fraction, sigma_optics)[0])
        if kw is None or not kw.any():
            d, ok = _edge_offsets(gray, q_img - origin, n_img, half)
        else:
            d, ok = _edge_offsets_windowed(gray, q_img - origin, n_img, half, kw)
        used = int(ok.sum())
        if used < 12:
            return Refinement(corners, used, len(pts), float("nan"), 0.0, False, "too few usable edges")

        # measurement and its normal, both carried back to the ideal plane
        m_id, _ = projection.undistort_newton(K, D, q_img[ok] + n_img[ok] * d[ok][:, None])
        nn_id, _ = projection.undistort_newton(K, D, q_img[ok] + n_img[ok] * 0.5)
        n_id = nn_id - q_id[ok]
        ln = np.linalg.norm(n_id, axis=1, keepdims=True)
        n_id = n_id / np.where(ln < 1e-15, 1.0, ln)

        H, rms = _fit_homography_to_lines(H, pts[ok], m_id, n_id, cauchy_c)
        if H is None:
            return Refinement(corners, used, len(pts), float("nan"), 0.0, False, "fit failed")

    out = projection.distort_points(K, D, _apply(H, quad)) - origin
    shift = float(np.mean(np.linalg.norm(out - corners, axis=1)))
    if not np.isfinite(out).all() or shift > cell_px:
        return Refinement(corners, used, len(pts), rms, shift, False, "diverged")
    # quality: one unweighted evaluation pass at the result, in px of `gray` (see `edge_residuals`)
    d, ok = edge_residuals(gray, out, marker_id, calib, origin=origin, dictionary=dictionary,
                           per_edge=per_edge, border_cells=border_cells)
    e_rms, dw = edge_quality(d, ok)
    return Refinement(out, used, len(pts), rms, shift, True, "", e_rms, dw)


def _H_from_corners(corners_img, calib, border_cells):
    from ..geometry import projection
    lo = 1.0 - border_cells
    hi = CELLS - 1.0 + border_cells
    quad = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], np.float64)
    ideal_c, _ = projection.undistort_newton(calib.K, calib.D, np.asarray(corners_img, np.float64))
    return cv2.getPerspectiveTransform(quad.astype(np.float32), ideal_c.astype(np.float32)), quad


def crop_origin(centre, w: int, h: int, crop: int) -> np.ndarray:
    """Top-left of the `crop` x `crop` window around `centre` (px), clamped into a `w` x `h` image: the crop
    `refine_cache.run` refines from."""
    x0 = int(max(0, min(round(centre[0] - crop / 2), w - crop)))
    y0 = int(max(0, min(round(centre[1] - crop / 2), h - crop)))
    return np.array([x0, y0], np.float64)
