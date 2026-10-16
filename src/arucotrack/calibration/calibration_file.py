"""The calibration file: what it must contain and how it is checked.

Format (`docs/CALIBRATION_FILE.md`): one JSON object

    {
      "camera_id": "cam1",                       optional
      "model": "FULL_OPENCV",                    pinhole: FULL_OPENCV | OPENCV | pinhole | k1k2p1p2k3 | k1k2 | k1k2p1p2 | rational
                                                 fisheye (Kannala-Brandt), read and refused: OPENCV_FISHEYE | fisheye
      "image_size": [2592, 1944],                width, height in pixels
      "intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...,      or  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                     "dist": [k1, k2, p1, p2, k3, ...]},              OpenCV order (fisheye: [k1, k2, k3, k4])
      "extrinsics": {"rvec": [..3], "tvec": [..3]}                    optional (or "R" (3x3) and "t" (3))
    }

The extrinsics are `cam_from_world` (OpenCV `solvePnP` convention: X_cam = R X_world + t), the world frame has z up
and the floor at z = 0, units metres. Without extrinsics the pipeline writes positions in the camera frame.

`check(doc)` returns the list of problems (empty = usable). `camera.Calib.from_json` refuses a file with problems.
The file is written by hand or by a few lines of Python from OpenCV results (docs/CALIBRATION.md).
"""
from __future__ import annotations

from typing import Any

import numpy as np

PINHOLE_MODELS = ("full_opencv", "opencv", "pinhole", "k1k2p1p2k3", "k1k2", "k1k2p1p2", "rational")
FISHEYE_MODELS = ("opencv_fisheye", "fisheye")
PINHOLE_DIST_LENGTHS = (4, 5, 8, 12, 14)
"""Accepted pinhole coefficient counts. 12 and 14 only with the thin-prism and tilt terms (coefficients 9 to 14)
zero: `projection.undistort_newton` inverts k1..k6, p1 and p2 only."""
MAX_IMAGE_SIDE = 65536
MODEL_NAMES = ("FULL_OPENCV", "OPENCV_FISHEYE")    # the model names the documentation uses


def model_name(doc: dict) -> str:
    """The file's `model`, lower case ("" when it states none)."""
    return str(doc.get("model") or (doc.get("intrinsics") or {}).get("model") or "").lower()


def is_fisheye(doc: dict) -> bool:
    """The file states a fisheye (Kannala-Brandt) model. The model is read from the file only."""
    return model_name(doc) in FISHEYE_MODELS


def intrinsic_matrix(doc: dict) -> np.ndarray:
    I = doc["intrinsics"]
    if all(k in I for k in ("fx", "fy", "cx", "cy")):
        return np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1]], np.float64)
    return np.asarray(I["K"], np.float64).reshape(3, 3)


def pose(doc: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(rvec (3,), tvec (3, 1)) of `extrinsics`, or None when the file has none."""
    import cv2
    E = doc.get("extrinsics")
    if not E:
        return None
    if E.get("rvec") is not None:
        rvec = np.asarray(E["rvec"], np.float64).reshape(3)
    else:
        rvec = cv2.Rodrigues(np.asarray(E["R"], np.float64).reshape(3, 3))[0].reshape(3)
    t = E.get("tvec") if E.get("tvec") is not None else E.get("t")
    return rvec, np.asarray(t, np.float64).reshape(3, 1)


def check(doc: Any) -> list[str]:
    """Problems that make the file unusable, in words ([] when it is fine)."""
    if not isinstance(doc, dict):
        return ["the file is not a JSON object"]
    errs: list[str] = []
    model = model_name(doc)
    if not model:
        errs.append(f"model is missing: state the camera model ({', '.join(MODEL_NAMES)})")
    elif model not in PINHOLE_MODELS + FISHEYE_MODELS:
        errs.append(f"model {doc.get('model')!r} is not one of {sorted(set(PINHOLE_MODELS + FISHEYE_MODELS))}")
    size = doc.get("image_size") or (doc.get("intrinsics") or {}).get("image_size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= MAX_IMAGE_SIDE for v in size)):
        errs.append(f"image_size must be [width, height], two integers in 1..{MAX_IMAGE_SIDE} pixels")
    I = doc.get("intrinsics")
    if not isinstance(I, dict):
        return errs + ["intrinsics is missing"]
    try:
        K = intrinsic_matrix(doc)
        if not (np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0 and np.allclose(K[2], [0, 0, 1])):
            errs.append("intrinsics: fx and fy must be positive and the last row of K must be [0, 0, 1]")
    except (KeyError, ValueError, TypeError):
        errs.append("intrinsics needs fx, fy, cx, cy or a 3x3 K")
    try:
        dist = np.asarray(I["dist"], np.float64).ravel()
        n_ok = (4,) if is_fisheye(doc) else PINHOLE_DIST_LENGTHS
        if len(dist) not in n_ok or not np.isfinite(dist).all():
            errs.append(f"intrinsics.dist has {len(dist)} coefficients, expected {' or '.join(map(str, n_ok))} "
                        f"({'fisheye k1..k4' if is_fisheye(doc) else 'OpenCV order k1 k2 p1 p2 [k3 ...]'})")
        elif not is_fisheye(doc) and np.any(dist[8:] != 0):
            errs.append("intrinsics.dist has non-zero thin-prism or tilt terms (coefficients 9 to 14), which this "
                        "package cannot invert. Calibrate with OpenCV's rational model (8 coefficients).")
    except (KeyError, ValueError, TypeError):
        errs.append("intrinsics.dist is missing")
    E = doc.get("extrinsics")
    if E:
        if not isinstance(E, dict):
            errs.append("extrinsics must be an object with rvec/tvec or R/t")
        else:
            try:
                if E.get("rvec") is None:
                    R = np.asarray(E["R"], np.float64).reshape(3, 3)
                    if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0):
                        errs.append("extrinsics.R is not a rotation (orthonormal, determinant +1)")
                elif np.asarray(E["rvec"], np.float64).size != 3:
                    errs.append("extrinsics.rvec must have 3 values")
                t = E.get("tvec") if E.get("tvec") is not None else E["t"]
                if np.asarray(t, np.float64).size != 3 or not np.isfinite(np.asarray(t, np.float64)).all():
                    errs.append("extrinsics.tvec (or t) must have 3 finite values")
            except (KeyError, ValueError, TypeError):
                errs.append("extrinsics needs rvec (or R, 3x3) and tvec (or t)")
    return errs
