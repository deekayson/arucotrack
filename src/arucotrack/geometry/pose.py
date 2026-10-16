"""Per-detection geometry from a detection cache, without decoding the video again (the paper, Section 3.6,
"Pose, trajectories and output").

Takes a calibration (`calibration.camera.Calib`) and a detection or refined cache (`.npz`), and writes a sibling `.geom.npz` with the marker centre in world coordinates, image radius, apparent side
and tilt for every detection.

Two rules: the square is posed with `SOLVEPNP_IPPE_SQUARE`, and only the marker centre is used from the pose.
The two IPPE solutions differ mainly in rotation, so a point transformed through the estimated rotation is not
trustworthy.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .. import provenance
from ..calibration.camera import Calib, check_calib_for_cache

# Rigid-square residual per pixel of isotropic corner noise, measured on one
# calibration at about 20 deg tilt and an 80-100 px marker: residual = 0.44 * noise
# (0.1 px -> 0.0457, 0.5 -> 0.2155, 1.0 -> 0.4371, 2.0 -> 0.8225). Used only to
# express the logged residual in the more legible units of corner noise. The
# factor depends on the geometry, so the conversion is approximate.
RESIDUAL_PER_PX_NOISE = 0.44
def pose_world(calib: Calib, corners: np.ndarray):
    """Marker centres in world coordinates via IPPE_SQUARE, with both solutions.

    Returns `xyz`, `tilt_deg`, `ok`, `residual`, `err_ratio` and `xyz_alt` (the centre of the second solution).
    `solvePnPGeneric` returns both IPPE solutions with their errors, which gives two diagnostics:

    `residual`: the reprojection error of the winning pose. With the marker
    size and the intrinsics known, four corners are 8 measurements for 6 pose
    parameters, so 2 degrees of freedom are left over and the residual is real
    information, available on every detection with no ground truth at all.
    Measured on one calibration, it is very close to linear in corner noise
    (0.1 px noise -> 0.046 px residual, 1.0 -> 0.437), so it is an
    annotation-free corner-noise meter.

    But it is blind to a uniform bias: shrink all four corners inward by
    1.9 px and the residual moves only to 0.024 px while the recovered height
    drops 38 mm, because a uniformly shrunk square is still the image of a
    square, merely a more distant one. A single camera cannot tell a smaller
    marker from a more distant one (the marker-size error of the paper,
    Section 4.2, "ArUcoTrack approach"). So the residual measures *noise*,
    never *scale*, and is no evidence about a scale bias.

    `err_ratio`: the second solution's error over the first's, an
    ambiguity flag. It was measured at about 25x at 0.2 px noise, falling to
    about 1.7x at 2 px, where the wrong solution starts winning. The ambiguity
    was measured to cost only single-digit mm of height, so this is a
    diagnostic, not a correction.
    """
    corners = np.asarray(corners, np.float64).reshape(-1, 4, 2)
    obj = calib.obj()
    n_det = len(corners)
    xyz = np.full((n_det, 3), np.nan)
    tilt = np.full(n_det, np.nan)
    residual = np.full(n_det, np.nan)
    err_ratio = np.full(n_det, np.nan)
    xyz_alt = np.full((n_det, 3), np.nan)
    ok_all = np.zeros(n_det, bool)
    for i, q in enumerate(corners):
        try:
            n, rvs, tvs, errs = cv2.solvePnPGeneric(obj, q.reshape(4, 1, 2), calib.K, calib.D,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            continue
        if not n:
            continue
        e = np.asarray(errs, np.float64).ravel()
        xyz[i] = (calib.Rt @ (tvs[0].reshape(3, 1) - calib.tvec)).ravel()
        Rm, _ = cv2.Rodrigues(rvs[0])
        nrm = Rm[:, 2]
        tilt[i] = float(np.degrees(np.arccos(min(1.0, abs(nrm[2]) / np.linalg.norm(nrm)))))
        residual[i] = float(e[0])
        if n > 1:
            xyz_alt[i] = (calib.Rt @ (tvs[1].reshape(3, 1) - calib.tvec)).ravel()
            err_ratio[i] = float(e[1] / max(e[0], 1e-12))
        ok_all[i] = True
    return xyz, tilt, ok_all, residual, err_ratio, xyz_alt


def radius(calib: Calib, uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    return np.linalg.norm(uv - np.array([calib.cx, calib.cy]), axis=1) / calib.half_diag


def side_px(corners: np.ndarray) -> np.ndarray:
    c = np.asarray(corners, np.float64).reshape(-1, 4, 2)
    return np.mean(np.linalg.norm(c - np.roll(c, -1, axis=1), axis=2), axis=1)


def corner_handling(z, in_prov: dict, name: str) -> dict:
    """The provenance note on the corners of a cache about to be posed.

    Every cache is posed in the OpenCV pixel-centre convention. A refined cache (it has `det_refine_rms_px`)
    records that convention as `corner_convention` in its provenance (`refine_cache.run`). One without the key does
    not say which convention its corners are in, so it is refused (ValueError). A per-row flag this function does not
    apply (`det_refine_used` False) is warned about: such rows are posed like refined ones."""
    note: dict = {"corner_convention": in_prov.get("corner_convention"), "unused_flags": []}
    if "det_refine_rms_px" in z.files and in_prov.get("corner_convention") is None:
        raise ValueError(f"{name}: a refined cache without corner_convention in its provenance, so the pixel "
                         f"convention of its corners is unknown; refine it again with this version")
    key = "det_refine_used"
    if key in z.files and z[key].dtype == bool and (~z[key]).any():
        note["unused_flags"].append(key)
        logger.warning("{}: {} flags rows (gate-failed rows resolved against their neighbours, see "
                       "det_gate_resolution) that derive does not filter; they are posed like refined rows", name, key)
    return note


def derive(cache_npz: str | Path, calib: Calib | None = None, out: Path | None = None) -> Path:
    """Per-detection geometry of a detection cache. A refined cache must record its corner convention
    (`corner_handling`)."""
    started = _dt.datetime.now(_dt.timezone.utc)
    if calib is None:
        raise ValueError(f"{Path(cache_npz).name}: no calibration given")
    check_calib_for_cache(cache_npz, calib)
    cache_npz = Path(cache_npz)
    from ..detection.detect import load_arrays
    z = load_arrays(cache_npz)
    in_prov = json.loads(str(z["provenance"]))
    corner_note = corner_handling(z, in_prov, cache_npz.name)
    corners = z["det_corners"].astype(np.float64)
    xyz, tilt, ok, residual, err_ratio, xyz_alt = pose_world(calib, corners)
    centres = corners.mean(1)
    prov = provenance.block(
        relpath=in_prov["input"]["relpath"], decoder=in_prov["environment"]["decoder"],
        pixel_pipeline=in_prov["environment"]["pixel_pipeline"],
        params={"derived_from": cache_npz.name, "detector": in_prov["parameters"],
                "calibration": {"source": calib.source, "sha256": calib.sha256, "marker_m": calib.marker_m,
                                "has_extrinsics": calib.has_extrinsics,
                                "frame": ("world: cam_from_world from the published extrinsics"
                                          if calib.has_extrinsics else
                                          "CAMERA-FRAME: the calibration has no camera pose, so the coordinates "
                                          "are in the camera's own frame and are stored as `det_xyz_cam`"),
                                "pnp": "SOLVEPNP_IPPE_SQUARE via solvePnPGeneric, centre only; both solutions kept"},
                "residual_note": "rigid-square residual: a corner-noise meter, BLIND to uniform scale bias",
                "corners": corner_note},
        started=started)
    out = out or cache_npz.with_name(cache_npz.name.replace(".npz", ".geom.npz"))
    # A calibration with intrinsics but no camera pose gives camera-frame
    # coordinates. They are stored under their own name, so a consumer that
    # needs world coordinates (`det_xyz`) fails loudly rather than reading the
    # wrong frame.
    xyz_key = "det_xyz" if calib.has_extrinsics else "det_xyz_cam"
    if not calib.has_extrinsics:
        logger.warning("{}: the calibration has no camera pose - writing camera-frame `det_xyz_cam`, "
                       "not world `det_xyz` ({})", cache_npz.name, calib.source)
    np.savez_compressed(out, det_seq=z["det_seq"], det_pts=z["det_pts"], det_id=z["det_id"],
                        det_centre=centres.astype(np.float32), det_tilt=tilt, det_ok=ok,
                        det_radius=radius(calib, centres), det_side=side_px(corners),
                        det_residual=residual, det_err_ratio=err_ratio, det_xyz_alt=xyz_alt,
                        frame=np.array("world" if calib.has_extrinsics else "camera"),
                        provenance=np.array(json.dumps(prov)), **{xyz_key: xyz})
    fin = ok & np.isfinite(xyz[:, 2])
    if not len(corners):
        # A capture with no detections at all is a finding, not an error: an eye-level camera can see
        # essentially nothing.
        logger.info("{}: 0 detections -> {}", cache_npz.name, out.name)
        return out
    rad = radius(calib, centres)
    med_res = float(np.nanmedian(residual)) if np.isfinite(residual).any() else float("nan")
    logger.info("{}: {} detections, {} posed, z median {:.3f} m, radius {:.2f}-{:.2f}, "
                "residual {:.4f} px (~{:.2f} px corner noise), err ratio {:.1f} -> {}",
                cache_npz.name, len(corners), int(fin.sum()), float(np.median(xyz[fin, 2])) if fin.any() else float("nan"),
                float(rad.min()), float(rad.max()), med_res, med_res / RESIDUAL_PER_PX_NOISE,
                float(np.nanmedian(err_ratio)) if np.isfinite(err_ratio).any() else float("nan"), out.name)
    return out
