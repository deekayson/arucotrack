"""The quality gate of the corner refinement (the paper, Section 3.5 "Quality gate"): when a converged refinement
is used (`gate`), and what a refused one becomes (`resolve_gated`, `gate.mode`).

In `resolve` mode `det_gate_resolution` records which corners each row of the refined cache holds:

    1  refined, the gate passed
    2  gate failed, the detector corners agree with the accepted neighbours on both sides (within gate.agree_px)
    3  gate failed, the refused refinement agrees with them
    4  unverified: accepted neighbours on one side only, or none (kept here, left out of the trajectory files)
    5  inadmissible: neighbours on both sides, neither candidate agrees (dropped, kept in `inadmissible_*`)

In `drop` mode every gate-failed row is removed.
"""
from __future__ import annotations

import numpy as np

MAX_EDGE_RMS_PX = 0.8
"""Gate: largest edge residual (rms px, image the refinement ran on) of a refinement that is used.
Measured on 600 random detections per camera. On a global-shutter camera: p99 0.565, maximum 0.635 (0.674 over the
20 detections with the largest ideal-plane residual, all blurred but on the marker). On an undistorted (zero-distortion
pinhole) view of a fisheye camera, made outside this package: p99 0.649, 599 of 600 at or below 0.8. Fits that are visibly wrong (detector seed off by a cell on one side,
motion-smeared cells) sit at 0.9-1.3."""
MAX_SHIFT_FRAC = 0.07
MIN_SHIFT_PX = 1.5
"""Gate: a refinement may move the corners by at most `max(MIN_SHIFT_PX, MAX_SHIFT_FRAC x diagonal)`
(mean corner shift, diagonal = mean of the detector quad's two diagonals). The fraction holds for both camera types
it was measured on. On the undistorted view of a fisheye camera the detector puts corners 10-15 px out on large
markers, and the refinement corrects them (39 of the 40 largest shifts were correct, shift/diagonal up to 0.074). At
0.07, 0 of 10709 rows of a global-shutter camera fail and 1 of 19408 rows of that view do. The global-shutter
camera's p99 (0.041) would reject 14 % of the rows of that view, almost all of them correct."""
MAX_DOWNWEIGHTED_FRAC = 0.3
"""Gate: largest share of interior edge samples unusable or > `refine.FAR_PX` from their line (measured over 600
rows per camera: p99 0.08, maximum 0.13 on a global-shutter camera, p99 0.17, maximum 0.30 on the undistorted view
of a fisheye camera). Do not relax it on the strength of the edge rms: the edge residual saturates at the profile half-window
(<= 4 px). Edges further off are not measured and land only here (`refine.edge_residuals`), so a gross misfit can
show a modest edge rms."""
GATE_BITS = {"edge_rms": 1, "shift": 2, "downweighted": 4}


def gate(edge_rms_px: float, shift_px: float, diag_px: float, downweighted_frac: float, *,
         max_edge_rms_px: float = MAX_EDGE_RMS_PX, max_shift_frac: float = MAX_SHIFT_FRAC,
         min_shift_px: float = MIN_SHIFT_PX, max_downweighted_frac: float = MAX_DOWNWEIGHTED_FRAC) -> int:
    """0 when a converged refinement may be used, else the OR of the `GATE_BITS` it fails (NaN fails)."""
    bits = 0
    if not edge_rms_px <= max_edge_rms_px:
        bits |= GATE_BITS["edge_rms"]
    if not shift_px <= max(min_shift_px, max_shift_frac * diag_px):
        bits |= GATE_BITS["shift"]
    if not downweighted_frac <= max_downweighted_frac:
        bits |= GATE_BITS["downweighted"]
    return bits


GATE_AGREE_PX = 3.0
"""Resolution of a gate-failed row: its corners are used only if their largest corner distance to the position
predicted from the neighbouring accepted rows of the same id is at most this (px of the image refined). A track
moves a few px per frame and the prediction interpolates, so 3 px is wide for true corners and narrow for a
wrong cell (a cell is 10-20 px on a marker 80-160 px across)."""
GATE_WINDOW_FRAMES = 5.0
"""Largest time from a gated row to an accepted neighbour used for its prediction, in nominal frame intervals."""
GATE_RESOLUTION = {1: "refined", 2: "detector", 3: "refused_refinement", 4: "unverified",
                   5: "inadmissible"}
UNVERIFIED = 4
INADMISSIBLE = 5
GATE_MODES = ("resolve", "drop")
"""`refine_cache.run(gate_mode=...)`. "resolve" (default) resolves gate-failed rows (`resolve_gated`). "drop"
keeps accepted refinements only: every gated row is removed from the per-row arrays, so any consumer,
`geometry.pose.derive` included, poses exactly those rows."""
DROP_CORNER_CONVENTION = ("pixel centre; REFINED on every row; gate-failed rows are removed; raw detector corners "
                          "in det_corners_raw, the refiner's output in det_corners_refined")


def resolve_gated(t, ids, detector, refined, accepted, gated, *, agree_px: float = GATE_AGREE_PX,
                  window: float = GATE_WINDOW_FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """`(code, agree_px)` per row: the `GATE_RESOLUTION` of every row and the distance that decided a gated one.

    Rows with `accepted` True are code 1, rows with neither flag are 0. A `gated` row looks for the
    nearest accepted row of the same id before it and after it, each within `window` nominal frame intervals of `t`
    (strictly earlier / later).

    - Neighbours on both sides: the prediction is linear in time between them. It is sharp (measured: accepted rows
      miss it by more than 3 px in 0.03-0.1 % of leave-one-out cases). The detector corners and the refused
      refinement are each compared with it by their largest corner distance. The nearer one is used (the detector
      on a tie) if within `agree_px` (code 2 / 3), else the row is inadmissible (5): both candidates are wrong
      against a reliable prediction.
    - Neighbour on one side only, or none: nothing can confirm the row. A single neighbour held still fails 84-97 %
      of good moving rows, so there is no one-sided prediction, and dropping such rows would discard mostly good
      ones. The row is kept as unverified (4), `agree` NaN. Its corners (the caller's choice, see `run`) are the
      refused refinement, which always converged to get here, and only otherwise the detector's. Whether it is
      used is decided by the consumer, not here.

    A neighbour must be strictly earlier / later (0 < dt): an accepted duplicate of the same (frame, id) is not
    one, since it sees the same imagery and would vouch for the row by repeating it, and a zero time step cannot
    be interpolated. A row without a finite time has no neighbours and is unverified. Detector and refined
    corners of a row share one corner order (the refiner starts from the detector quad and keeps its labelling),
    which is what makes the corner-wise distance valid.
    Inputs: `t` (n, nominal frame units, NaN = no time), `ids` (n), and `detector`, `refined` (n, 4, 2) in one
    image's px, the refined array holding the accepted rows' corners."""
    t = np.asarray(t, np.float64)
    ids = np.asarray(ids)
    det = np.asarray(detector, np.float64).reshape(-1, 4, 2)
    ref = np.asarray(refined, np.float64).reshape(-1, 4, 2)
    accepted = np.asarray(accepted, bool) & np.isfinite(t)
    gated = np.asarray(gated, bool)
    code = np.zeros(len(t), np.int8)
    code[accepted] = 1
    agree = np.full(len(t), np.nan)
    for i in np.unique(ids[gated]):
        acc = np.flatnonzero(accepted & (ids == i))
        acc = acc[np.argsort(t[acc], kind="stable")]
        ta = t[acc]
        for k in np.flatnonzero(gated & (ids == i)):
            code[k] = UNVERIFIED
            if not np.isfinite(t[k]):
                continue
            j = int(np.searchsorted(ta, t[k], side="left"))        # first accepted at or after t[k]
            lo = acc[j - 1] if j > 0 and 0 < t[k] - ta[j - 1] <= window else None
            j = int(np.searchsorted(ta, t[k], side="right"))       # first accepted strictly after t[k]
            hi = acc[j] if j < len(ta) and 0 < ta[j] - t[k] <= window else None
            if lo is not None and hi is not None:
                w = (t[k] - t[lo]) / (t[hi] - t[lo])
                pred = (1 - w) * ref[lo] + w * ref[hi]
            else:
                continue                                           # one-sided or none: unverified
            d_det = float(np.linalg.norm(det[k] - pred, axis=1).max())
            d_ref = float(np.linalg.norm(ref[k] - pred, axis=1).max())
            d, c = (d_det, 2) if d_det <= d_ref else (d_ref, 3)
            agree[k] = d
            code[k] = c if d <= agree_px else INADMISSIBLE
    return code, agree
