"""Per-row marker orientation (world-from-marker) for `<stem>.orient.npz`, which a viewer can use to draw the markers:
both IPPE solutions per refined detection, the one closer to the smoothed orientation chosen, and the quaternions
averaged over a time window (`window_average`). Quaternions are xyzw."""
from __future__ import annotations

import cv2
import numpy as np


def quat_from_R(R) -> np.ndarray:
    m = np.asarray(R, float)
    w, v = np.linalg.eigh(np.array([  # Bar-Itzhack/Markley K matrix: robust for any rotation
        [m[0, 0] - m[1, 1] - m[2, 2], m[1, 0] + m[0, 1], m[2, 0] + m[0, 2], m[2, 1] - m[1, 2]],
        [m[1, 0] + m[0, 1], m[1, 1] - m[0, 0] - m[2, 2], m[2, 1] + m[1, 2], m[0, 2] - m[2, 0]],
        [m[2, 0] + m[0, 2], m[2, 1] + m[1, 2], m[2, 2] - m[0, 0] - m[1, 1], m[1, 0] - m[0, 1]],
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1], m[0, 0] + m[1, 1] + m[2, 2]]]) / 3.0)
    q = v[:, np.argmax(w)]
    return q if q[3] >= 0 else -q


def geodesic(a, b) -> float:
    """Rotation angle (rad) between two unit quaternions (q and -q are the same rotation)."""
    return float(2 * np.arccos(min(1.0, abs(float(np.dot(a, b))))))


def window_average(q, t, half_s: float) -> np.ndarray:
    """Per row the principal-eigenvector average of the quaternions within +-half_s seconds (sign-invariant, the
    standard quaternion mean, which for a window of a few degrees equals the SLERP midpoint)."""
    q = np.asarray(q, float); t = np.asarray(t, float)
    out = np.empty_like(q)
    lo = np.searchsorted(t, t - half_s - 1e-9, side="left"); hi = np.searchsorted(t, t + half_s + 1e-9, side="right")
    for k in range(len(q)):
        s = q[lo[k]:hi[k]]
        w, v = np.linalg.eigh(s.T @ s)
        m = v[:, -1]
        out[k] = m if m[3] >= 0 else -m
    return out


def choose_and_smooth(cand, t, half_s: float = 0.1, iters: int = 4):
    """cand (N,2,4): IPPE's best solution first, the alternative second. Per row pick the solution closest to the smoothed
    orientation (iterated: smooth, re-pick, smooth). Returns (pick (N,) 0/1, smoothed (N,4), rejected (N,) bool = the
    alternative was picked). Rows must be one id, sorted by time."""
    cand = np.asarray(cand, float)
    n = len(cand)
    pick = np.zeros(n, int)
    for _ in range(iters):
        chosen = cand[np.arange(n), pick]
        ref = window_average(chosen, t, half_s)
        new = np.array([int(geodesic(cand[k, 1], ref[k]) < geodesic(cand[k, 0], ref[k])) for k in range(n)])
        if (new == pick).all():
            break
        pick = new
    chosen = cand[np.arange(n), pick]
    return pick, window_average(chosen, t, half_s), pick == 1


def candidates(calib, corners) -> tuple[np.ndarray, np.ndarray]:
    """(N,2,4) world-from-marker quaternions of both IPPE_SQUARE solutions (best first) and (N,2) reprojection errors.
    Marker frame: `calibration.camera.Calib.obj()` (x right, y up, z out of the marker face). `solvePnPGeneric` knows only the
    pinhole distortion model, so a fisheye calibration is refused (ValueError) rather than solved through the wrong
    model."""
    if calib.fisheye:
        raise ValueError(f"{calib.source}: marker orientations are solved with the pinhole model, and this "
                         f"calibration is a fisheye model")
    corners = np.asarray(corners, float).reshape(-1, 4, 2)
    obj = calib.obj()
    Rwc = calib.Rt                                   # camera -> world
    out = np.full((len(corners), 2, 4), np.nan); err = np.full((len(corners), 2), np.nan)
    for i, c in enumerate(corners):
        try:
            n, rv, tv, e = cv2.solvePnPGeneric(obj, c.reshape(4, 1, 2), calib.K, calib.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            continue
        for j in range(min(n, 2)):
            out[i, j] = quat_from_R(Rwc @ cv2.Rodrigues(rv[j])[0])
            err[i, j] = float(np.asarray(e).ravel()[j])
        if n == 1:
            out[i, 1], err[i, 1] = out[i, 0], err[i, 0]
    return out, err
