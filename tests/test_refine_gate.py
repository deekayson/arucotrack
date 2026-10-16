"""The refinement quality metric and gate.

`Refinement.rms_px` is an ideal-plane IRLS residual in units of 1/f and is ~0 on real caches, so it
cannot rank or gate anything. `edge_rms_px` re-measures every interior edge against the fitted lines
in image px. The failure it must catch: a detector quad one cell too wide on one side (the white
overhang taken for border), which the refiner cannot pull back.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from arucotrack.refinement import refine
from arucotrack.refinement import refine_cache


class _Canvas:
    """A pinhole calibration without distortion (D = 0)."""
    K = np.array([[750.0, 0.0, 300.0], [0.0, 750.0, 300.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)


def _render(mid: int, side: float = 150.0, rot: float = 0.3, overhang: float = 0.6, blur: float = 1.0):
    """Marker (one-cell border) inside a white overhang on a dark background, on a 600 px canvas.

    8x supersampled and area-averaged so the edges are exact. Returns the image and the TRUE outer corners
    of the black square in the centre convention."""
    S, cell = 8, 64
    g = refine.cell_grid(mid).astype(np.float32)
    o = int(round(overhang * cell)); pad = o + 2 * cell
    tex = np.full((8 * cell + 2 * pad,) * 2, 60, np.float32)
    tex[pad - o:pad + 8 * cell + o, pad - o:pad + 8 * cell + o] = 215
    tex[pad:pad + 8 * cell, pad:pad + 8 * cell] = cv2.resize(g, (8 * cell,) * 2, interpolation=cv2.INTER_NEAREST) * 185 + 25
    c = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    truth = c @ R.T + 300.0
    # warpPerspective puts pixel centres on the integers: an edge coordinate e is at e - 0.5
    src = np.array([[pad, pad], [pad + 8 * cell, pad], [pad + 8 * cell, pad + 8 * cell], [pad, pad + 8 * cell]],
                   np.float32) - 0.5
    H = cv2.getPerspectiveTransform(src, ((truth + 0.5) * S - 0.5).astype(np.float32))
    big = cv2.warpPerspective(tex, H, (600 * S, 600 * S), flags=cv2.INTER_LINEAR, borderValue=60)
    img = cv2.GaussianBlur(cv2.resize(big, (600, 600), interpolation=cv2.INTER_AREA), (0, 0), blur)
    img = img + np.random.default_rng(0).normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), truth


def _refine(img, seed_centre, mid):
    """`refine_cache`'s calling convention: centre-convention seed + 0.5, 224 px crop, result - 0.5."""
    c0 = seed_centre + 0.5
    o = np.round(c0.mean(0) - 112)
    crop = img[int(o[1]):int(o[1]) + 224, int(o[0]):int(o[0]) + 224]
    r = refine.refine(crop, c0 - o, mid, _Canvas(), origin=o, border_cells=1.0)
    diag = 0.5 * (np.linalg.norm(seed_centre[0] - seed_centre[2]) + np.linalg.norm(seed_centre[1] - seed_centre[3]))
    return r, r.corners + o - 0.5, diag


@pytest.mark.parametrize("mid", [19, 33])
def test_a_clean_canvas_marker_passes_the_gate(mid):
    img, truth = _render(mid)
    r, out, diag = _refine(img, truth, mid)
    assert r.converged
    assert np.abs(out - truth).max() < 0.1
    assert r.edge_rms_px < 0.1 and r.downweighted_frac == 0.0
    assert r.rms_px < 1e-3                       # the ideal-plane metric, units 1/f
    assert refine_cache.gate(r.edge_rms_px, r.shift_px, diag, r.downweighted_frac) == 0


@pytest.mark.parametrize("mid", [19, 33])
def test_a_seed_one_cell_too_wide_on_one_side_is_gated(mid):
    """A detector quad one cell too wide on one side: the detector took the white overhang for a border column. The refiner
    converges (its capture range is a few px) to a quad that is still ~14 px off - and only the edge
    residual says so, `rms_px` would have called it a good fit."""
    img, truth = _render(mid)
    seed = truth.copy()
    cell = (truth[1] - truth[0]) / 8
    seed[0] -= cell; seed[3] -= cell
    r, out, diag = _refine(img, seed, mid)
    assert r.converged                           # the convergence flag alone would accept it
    assert r.rms_px < 1e-3                       # ...and its metric is blind to the misfit
    assert np.abs(out - truth).max() > 5.0
    assert r.edge_rms_px > refine_cache.MAX_EDGE_RMS_PX
    bits = refine_cache.gate(r.edge_rms_px, r.shift_px, diag, r.downweighted_frac)
    assert bits & refine_cache.GATE_BITS["edge_rms"]
    assert bits & refine_cache.GATE_BITS["downweighted"]


def test_edge_residuals_of_a_detector_quad_rank_it_against_the_truth():
    img, truth = _render(20)
    o = np.round(truth.mean(0) + 0.5 - 112)
    crop = img[int(o[1]):int(o[1]) + 224, int(o[0]):int(o[0]) + 224]
    good = refine.edge_quality(*refine.edge_residuals(crop, truth + 0.5 - o, 20, _Canvas(), origin=o, border_cells=1.0))
    off = truth.copy(); off[1] += (truth[1] - truth.mean(0)) * 0.08          # one corner 8 % out, as a detector can misplace it
    bad = refine.edge_quality(*refine.edge_residuals(crop, off + 0.5 - o, 20, _Canvas(), origin=o, border_cells=1.0))
    assert good[0] < 0.1 and bad[0] > 0.5


def test_the_gate_rules():
    g = refine_cache.gate
    B = refine_cache.GATE_BITS
    assert g(0.3, 1.0, 100.0, 0.0) == 0
    assert g(0.81, 1.0, 100.0, 0.0) == B["edge_rms"]
    assert g(0.3, 7.1, 100.0, 0.0) == B["shift"]           # 0.07 x 100
    assert g(0.3, 6.9, 100.0, 0.0) == 0
    assert g(0.3, 1.4, 10.0, 0.0) == 0                     # the 1.5 px floor
    assert g(0.3, 1.0, 100.0, 0.31) == B["downweighted"]
    assert g(float("nan"), float("nan"), 100.0, float("nan")) == B["edge_rms"] | B["shift"] | B["downweighted"]
    assert g(0.9, 1.0, 100.0, 0.0, max_edge_rms_px=1.0) == 0
