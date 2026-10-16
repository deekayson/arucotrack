"""The exposure fraction (exposure time / frame interval) of a camera, from its own video: blur length vs image speed.

A marker moving `v` px per frame is smeared during the exposure over a box `w = f |v . n|` px across an edge of normal
`n`, where `f` is the exposure fraction. Each measured row gives `w` per marker axis from the whole cell pattern. The
id's known 8x8 pattern is convolved with a box of free width `w` (the exposure) and a Gaussian of the camera's at-rest
optics sigma (fixed), and fitted along six scan lines through the data rows (`fit_scan`). The step responses are
analytic, so `w` may exceed a cell. The estimate is the median of `w / |v . n|` over axes with `|v . n| >= VMIN`, and
its uncertainty a bootstrap over markers.

Every input comes from the caller: the speed from the detector track (`refine_cache.detector_velocity`) and the
optics sigma from the configuration. On a camera whose exposure metadata gave 0.978, this estimator (sigma fixed at rest) read 1.033,
+5.6 % against the metadata. 50 fast rows give a statistical uncertainty of about +-1 %.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import optimize, special

from . import refine

VMIN = 6.0
"""px/frame across the scanned edges: below this the box is too short to measure against the optics."""
RMS_REL_MAX = 0.12
"""A scan fit whose rms exceeds this fraction of the fitted pattern contrast is not used."""
MIN_ROWS = 50
"""Fewest fast rows (rows with at least one accepted axis at `VMIN`) an estimate is made from."""


def _G(x, s):
    return x * special.ndtr(x / s) + s * np.exp(-0.5 * (x / s) ** 2) / math.sqrt(2 * math.pi)


def step_resp(x, w, s):
    """A unit step convolved with a box of width `w` and a Gaussian of sigma `s`."""
    w = max(w, 0.05)
    return (_G(x + w / 2, s) - _G(x - w / 2, s)) / w


def _quad_cells(b):
    lo, hi = 1.0 - b, 7.0 + b
    return np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], np.float64)


def _H(corners, b):
    """The homography from cell coordinates to image px, from outer corners in the refiner's pixel-EDGE convention."""
    return cv2.getPerspectiveTransform(_quad_cells(b).astype(np.float32),
                                       np.asarray(corners, np.float64).reshape(4, 2).astype(np.float32))


def _apply(H, p):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def _family_axes(H):
    """Marker centre and the unit image normals of the u-lines (x = const) and the v-lines (y = const) there."""
    c = _apply(H, [[4, 4]])[0]
    tu = _apply(H, [[4, 4.01]])[0] - c
    tv = _apply(H, [[4.01, 4]])[0] - c
    nu = np.array([-tu[1], tu[0]]); nu /= np.linalg.norm(nu)
    nv = np.array([-tv[1], tv[0]]); nv /= np.linalg.norm(nv)
    return c, nu, nv


def _scan_lines(img, H, g, b, fam, axis, centre, step=1.0):
    """Six scan lines through the data rows along `fam`: per line the samples s (px along `axis` from `centre`), their
    intensities, and the pattern transitions (position and step height, the two outer ones marked 'L' / 'R')."""
    lo, hi = 1.0 - b, 7.0 + b
    lines = []
    for i in range(1, 7):
        y = i + 0.5
        xs = np.arange(0.2, 7.8 + 1e-9, 0.02)
        pts = np.c_[xs, np.full_like(xs, y)] if fam == "u" else np.c_[np.full_like(xs, y), xs]
        sref = (_apply(H, pts) - centre) @ axis
        sign = -1.0 if sref[-1] < sref[0] else 1.0
        s_samp = np.arange(min(sref[0], sref[-1]), max(sref[0], sref[-1]), step)
        order = np.argsort(sref)
        xs_of_s = np.interp(s_samp, sref[order], xs[order])
        pts_s = np.c_[xs_of_s, np.full_like(xs_of_s, y)] if fam == "u" else np.c_[np.full_like(xs_of_s, y), xs_of_s]
        val = refine._bilinear(img, _apply(H, pts_s))
        ok = np.isfinite(val)
        row = g[i, :] if fam == "u" else g[:, i]
        xe, he = [lo], [("L",)]
        for j in range(1, 8):
            if row[j] != row[j - 1]:
                xe.append(float(j)); he.append(float(row[j]) - float(row[j - 1]))
        xe.append(hi); he.append(("R",))
        pe = np.c_[xe, np.full(len(xe), y)] if fam == "u" else np.c_[np.full(len(xe), y), xe]
        se = (_apply(H, pe) - centre) @ axis
        lines.append(dict(s=s_samp[ok], v=val[ok], se=se, he=he, sign=sign))
    return lines


def _pack(lines):
    E = max(len(L["se"]) for L in lines)
    S, SE, KIND, HH, SG = [], [], [], [], []
    for L in lines:
        n = len(L["s"])
        se = np.zeros(E); kind = np.full(E, -1); hh = np.zeros(E)
        for j, (x, h) in enumerate(zip(L["se"], L["he"])):
            se[j] = x
            if isinstance(h, tuple):
                kind[j] = 1 if h[0] == "L" else 2
            else:
                kind[j] = 0; hh[j] = h
        S.append(L["s"]); SE.append(np.tile(se, (n, 1))); KIND.append(np.tile(kind, (n, 1)))
        HH.append(np.tile(hh, (n, 1))); SG.append(np.full(n, L["sign"]))
    return dict(s=np.concatenate(S), se=np.concatenate(SE), kind=np.concatenate(KIND), hh=np.concatenate(HH),
                sign=np.concatenate(SG))


def _model(p, P):
    """The blurred pattern on every scan line. Steps are taken in increasing s: where the cell index runs against s
    (sign < 0) the steps are negated and the far end in s is the high-index outside (oR)."""
    c, k, w, sig, a, bb, oL, oR = p
    hx = np.where(P["kind"] == 0, P["hh"], np.where(P["kind"] == 1, -oL, np.where(P["kind"] == 2, oR, 0.0)))
    hs = hx * P["sign"][:, None]
    base = np.where(P["sign"] > 0, oL, oR)
    m = base + np.sum(hs * step_resp(P["s"][:, None] - (c + k * P["se"]), w, sig), axis=1)
    return a + bb * m


def fit_scan(img, corners, g, b, fam, axis, centre, w0, sigma):
    """Box width `w` (px) of the pattern along `axis` for line family `fam`, optics sigma fixed, with the fit's
    translation `c`, scale `k`, rms and contrast. `corners` (edge convention) only place the pattern: `c` (+-12 px)
    and `k` (0.85-1.15) are free. None when the scan has too few samples or no contrast."""
    H = _H(corners, b)
    lines = _scan_lines(img, H, g, b, fam, axis, centre)
    y = np.concatenate([L["v"] for L in lines])
    if len(y) < 60:
        return None
    lo_v, hi_v = np.percentile(y, 3), np.percentile(y, 97)
    if hi_v - lo_v < 15:
        return None
    P = _pack(lines)

    def res(p):
        q = p.copy(); q[3] = sigma
        return _model(q, P) - y
    p0 = np.array([0.0, 1.0, max(w0, 0.5), sigma, lo_v, hi_v - lo_v, 0.6, 0.6])
    lb = [-12, 0.85, 0.0, 0.0, -60, 5, -0.5, -0.5]
    ub = [12, 1.15, 40.0, max(4.0, sigma + 1.0), 255, 400, 1.6, 1.6]
    p0[3] = min(max(sigma, lb[3] + 1e-6), ub[3] - 1e-6)
    best = None
    for c0 in (0.0, -0.35 * max(w0, 1), 0.35 * max(w0, 1)):
        if best is not None and np.sqrt(np.mean(best.fun ** 2)) < RMS_REL_MAX * max(best.x[5], 1.0):
            break
        p0[0] = c0
        try:
            r = optimize.least_squares(res, p0, bounds=(lb, ub), x_scale=[1, 0.01, 1, 0.3, 10, 30, 0.2, 0.2],
                                       max_nfev=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    p = best.x
    rms = float(np.sqrt(np.mean(best.fun ** 2)))
    return dict(c=float(p[0]), k=float(p[1]), w=float(p[2]), rms=rms, contrast=float(p[5]),
                rms_rel=rms / max(float(p[5]), 1.0), n=int(len(y)))


def blur_widths(img, corners, marker_id: int, border_cells: float, velocity, sigma_optics: float,
                vmin: float = VMIN, dictionary=None) -> list[tuple[float, float]]:
    """`[(|v . n|, w), ...]` for the marker axes of one detection whose speed across them is at least `vmin` and
    whose scan fit is accepted (`RMS_REL_MAX`). `corners` is the detector quad in the refiner's pixel-EDGE
    convention, px of `img`, and `velocity` is in px/frame on the same grid. `dictionary` is the marker's
    `cv2.aruco` dictionary (None: `refine.cell_grid`'s default), which gives the cell pattern the fit models."""
    v = np.asarray(velocity, np.float64).reshape(2)
    if not np.isfinite(v).all():
        return []
    g = refine.cell_grid(int(marker_id), dictionary)
    try:
        centre, nu, nv = _family_axes(_H(corners, border_cells))
    except (cv2.error, ValueError, FloatingPointError):
        return []
    out = []
    for fam, nn in (("u", nu), ("v", nv)):
        vn = abs(float(nn @ v))
        if not vn >= vmin:
            continue
        fs = fit_scan(img, corners, g, border_cells, fam, nn, centre, w0=vn, sigma=float(sigma_optics))
        if fs is not None and fs["rms_rel"] <= RMS_REL_MAX:
            out.append((vn, fs["w"]))
    return out


def estimate_exposure_fraction(measurements, *, min_rows: int = MIN_ROWS, vmin: float = VMIN, n_boot: int = 300,
                               seed: int = 0) -> dict:
    """The exposure fraction from `measurements` (dicts with `row`, `marker_id`, `vn`, `w`): the median of `w / vn`
    over measurements with `vn >= vmin`. Returns a dict: `ok`, `value`, `sd` (bootstrap over markers when there are
    at least 3, else the median's standard error from the rows), `sd_method`, `n_rows` (distinct fast rows),
    `n_measurements`, `n_markers`, the Theil-Sen slope of w on vn with its 95 % interval (a cross-check that
    allows an intercept), and `reason` when not ok (fewer than `min_rows` fast rows)."""
    from scipy.stats import theilslopes
    fast = [m for m in measurements if float(m["vn"]) >= vmin and np.isfinite(m["w"])]
    n_rows = len({m["row"] for m in fast})
    out = dict(ok=False, value=None, sd=None, sd_method=None, n_rows=n_rows, n_measurements=len(fast),
               n_markers=len({m["marker_id"] for m in fast}), vmin_px_per_frame=vmin, min_rows=min_rows,
               estimator="median of box width / speed across the edges, optics sigma fixed", theil_slope=None,
               theil_ci95=None, reason=None)
    if n_rows < min_rows:
        out["reason"] = f"{n_rows} fast rows (>= {vmin:g} px/frame across the edges, scan fit accepted); at least " \
                        f"{min_rows} are needed"
        return out
    ratio = np.array([m["w"] / m["vn"] for m in fast], float)
    mids = np.array([m["marker_id"] for m in fast])
    out["value"] = float(np.median(ratio))
    um = np.unique(mids)
    if len(um) >= 3:
        rng = np.random.default_rng(seed)
        bs = [np.median(np.concatenate([ratio[mids == m] for m in rng.choice(um, size=len(um), replace=True)]))
              for _ in range(n_boot)]
        out["sd"], out["sd_method"] = float(np.std(bs)), f"bootstrap over markers ({n_boot})"
    else:
        mad_sd = 1.4826 * float(np.median(np.abs(ratio - np.median(ratio))))
        out["sd"], out["sd_method"] = 1.2533 * mad_sd / math.sqrt(len(ratio)), "median standard error over rows"
    vn = np.array([m["vn"] for m in measurements], float)
    w = np.array([m["w"] for m in measurements], float)
    if len(vn) >= 3 and np.ptp(vn) > 0:
        ts = theilslopes(w, vn)
        out["theil_slope"], out["theil_ci95"] = float(ts[0]), [float(ts[2]), float(ts[3])]
    out["ok"] = True
    return out
