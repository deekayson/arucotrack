"""Refine every detection's corners against the imagery, into a cache of its own (the paper, Sections 3.4
"Corner refinement" and 3.5 "Quality gate").

Raw detector corners give a noisier marker pose with a scale bias that depends on the detector settings, so the pose
chain is fed refined corners. The output has the schema of the detection cache, and `geometry.pose.derive` reads
either.
The video is decoded once, straight through, with the detections grouped by frame, and each marker is refined in a
square crop (`refine.crop_px`, 224 px by default, enough for a marker up to about 155 px across).

A refinement that does not converge drops its detection: mixing raw and refined corners in one track would mix two
corner conventions. A converged refinement that fails the quality gate (`gate.gate`) is handled by `gate.mode`
(`gate.resolve_gated`, the codes of `det_gate_resolution` in `gate`). The raw detector corners stay beside every row
in `det_corners_raw`.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np
from loguru import logger

from .. import provenance
from ..calibration.camera import check_calib_for_cache
from ..detection import detect
from ..io.video import decoder_label, frames, probe
from . import exposure_blur, refine
from .gate import (DROP_CORNER_CONVENTION, GATE_AGREE_PX, GATE_BITS, GATE_MODES, GATE_RESOLUTION, GATE_WINDOW_FRAMES,
                   INADMISSIBLE, MAX_DOWNWEIGHTED_FRAC, MAX_EDGE_RMS_PX, MAX_SHIFT_FRAC, MIN_SHIFT_PX, UNVERIFIED, gate,
                   resolve_gated)

CROP = 224


def _source(video: Path | None) -> Path:
    """The video this cache came from, as the caller names it (`video`). Nothing is looked up."""
    if video is None:
        raise ValueError("refinement reads the cache's video: pass its path (video=)")
    if not Path(video).is_file():
        raise FileNotFoundError(f"video {video} does not exist")
    return Path(video)


def _detect_sha256(cache_npz: Path) -> tuple[str, str]:
    """(sha256 of the detect npz, where it came from). The writer's sidecar records `npz_sha256`
    (`detect._write_cache`). It is trusted when the sidecar is not older than the npz (a kill between the two
    renames leaves an older sidecar beside a newer npz), else the npz is hashed here, once, in chunks."""
    cache_npz = Path(cache_npz)
    side = cache_npz.with_suffix(".json")
    try:
        sha = json.loads(side.read_text()).get("npz_sha256")
        if sha and side.stat().st_mtime >= cache_npz.stat().st_mtime:
            return sha, "sidecar"
    except (OSError, ValueError):
        pass
    return provenance.sha256_file(cache_npz), "computed"


def resolve_border_cells(spec) -> tuple[float, dict]:
    """The border constant a refinement uses, and where it came from: `(value, block)`.

    Only what the caller passes counts (`marker.border_cells`). Nothing is discovered from the directory. A number
    (or a string that parses as one) is an `override`. Anything else is a ValueError. How to set the constant for a
    camera: docs/CALIBRATION.md, "Border constant".

    The block is what the refined provenance records: `border_cells` and `border_source`."""
    try:
        v = float(spec)
    except (TypeError, ValueError):
        raise ValueError(f"border_cells {spec!r} is not a number") from None
    return v, {"border_cells": v, "border_source": "override"}


MAX_NEIGHBOUR_GAP = 2.5
"""Largest time from a detection to the neighbour its velocity uses, in nominal frame intervals: a single dropped
frame (2 intervals) still pairs, two (3 intervals) do not."""


def nominal_frame_period(pts) -> float:
    """The video's nominal frame interval (s) from its frame timestamps: the mean of the intervals within 0.5-1.5x
    their median. The median alone is not enough: timestamps stored to the millisecond make single 60 Hz intervals
    read 16 or 17 ms, and the median lands on one of them (1.7 % off). Dropped frames (2+ intervals) are excluded by
    the band."""
    d = np.diff(np.asarray(pts, np.float64))
    d = d[np.isfinite(d) & (d > 0)]
    if not len(d):
        raise ValueError("no increasing frame timestamps to take a frame period from")
    med = float(np.median(d))
    return float(d[(d >= 0.5 * med) & (d <= 1.5 * med)].mean())


def detector_velocity(seq, ids, corners, usable=None, *, t=None, period=None,
                      max_gap: float = MAX_NEIGHBOUR_GAP) -> tuple[np.ndarray, np.ndarray]:
    """Image velocity of every detection from the detector quad centres of the nearest earlier and later detection
    of the same id in time: `(v, kind)`. v (n, 2) is in px per NOMINAL frame (displacement / time difference x
    `period`, so a dropped frame changes nothing and the unit matches the exposure fraction). kind is 2 central
    (earlier and later), 1 one-sided (one of them), 0 none (NaN velocity). A neighbour more than `max_gap` nominal
    intervals away is not used. `t` is each row's timestamp (s, the cache's `det_pts`) and `period` the nominal
    interval (`nominal_frame_period`). Without them `seq` is the time and the period 1.

    A row whose (seq, id) occurs more than once has no velocity and is no neighbour (it cannot be paired with the
    right detection), nor is a row with non-finite corners or time, or `usable` False."""
    seq = np.asarray(seq, np.int64)
    ids = np.asarray(ids, np.int64)
    if (t is None) != (period is None):
        raise ValueError("detector_velocity: give both t and period, or neither")
    tt = seq.astype(np.float64) if t is None else np.asarray(t, np.float64)
    T = 1.0 if period is None else float(period)
    c = np.asarray(corners, np.float64).reshape(-1, 4, 2)
    ok = np.isfinite(c.reshape(len(c), -1)).all(1) & np.isfinite(tt)
    if usable is not None:
        ok &= np.asarray(usable, bool)
    cen = c.mean(1)
    count: dict = {}
    for s, i in zip(seq.tolist(), ids.tolist()):
        count[(s, i)] = count.get((s, i), 0) + 1
    good = np.array([count[(s, i)] == 1 for s, i in zip(seq.tolist(), ids.tolist())], bool) & ok
    v = np.full((len(seq), 2), np.nan)
    kind = np.zeros(len(seq), np.int8)
    lim = max_gap * T
    for i in np.unique(ids[good]):
        rows = np.flatnonzero(good & (ids == i))
        rows = rows[np.lexsort((seq[rows], tt[rows]))]
        for n, k in enumerate(rows):
            a = rows[n - 1] if n > 0 and 0 < tt[k] - tt[rows[n - 1]] <= lim else None
            b = rows[n + 1] if n + 1 < len(rows) and 0 < tt[rows[n + 1]] - tt[k] <= lim else None
            if a is not None and b is not None:
                v[k] = (cen[b] - cen[a]) / (tt[b] - tt[a]) * T; kind[k] = 2
            elif b is not None:
                v[k] = (cen[b] - cen[k]) / (tt[b] - tt[k]) * T; kind[k] = 1
            elif a is not None:
                v[k] = (cen[k] - cen[a]) / (tt[k] - tt[a]) * T; kind[k] = 1
    return v, kind


EXPOSURE_MAX_ROWS = 200
"""Fast rows sampled for the image-based exposure estimate (spread over ids, seeded): 200 rows were measured to give
about +-0.5 %, far inside the estimator's own systematic error."""


def video_exposure_estimate(*, src: Path, source: str, info, work: np.ndarray, seq: np.ndarray,
                            ids: np.ndarray, velocity: np.ndarray, usable: np.ndarray, border_cells: float,
                            sigma_optics: float, crop: int = CROP, max_frames: int | None = None,
                            max_rows: int = EXPOSURE_MAX_ROWS, min_rows: int = exposure_blur.MIN_ROWS,
                            vmin: float = exposure_blur.VMIN, seed: int = 0, dictionary=None) -> dict:
    """The exposure fraction of this video from its own fast rows (`exposure_blur`): a seeded sample of up to
    `max_rows` detections with a detector speed of at least `vmin` px/frame, spread over the ids (round robin), is
    decoded (up to the last sampled frame only), cropped exactly as the refinement crops, and measured with the
    detector quad as the pattern's starting place. Returns `exposure_blur.estimate_exposure_fraction`'s dict plus
    `n_candidates` and `n_sampled`, with `ok` False (and a `reason`) when fewer than `min_rows` fast rows were
    measured. `dictionary` is the markers' `cv2.aruco` dictionary (None: the default of `refine.cell_grid`), which
    gives the cell pattern the blur fit models."""
    speed = np.linalg.norm(np.where(np.isfinite(velocity), velocity, 0.0), axis=1)
    cand = np.flatnonzero(np.asarray(usable, bool) & np.isfinite(velocity).all(1) & (speed >= vmin))
    if max_frames is not None:
        cand = cand[seq[cand] < max_frames]
    rng = np.random.default_rng(seed)
    pools = {}
    for i in cand:
        pools.setdefault(int(ids[i]), []).append(int(i))
    for k in pools:
        rng.shuffle(pools[k])
    pick: list[int] = []
    while len(pick) < max_rows and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(pick) < max_rows:
                pick.append(pools[k].pop())
    by_frame: dict[int, list[int]] = {}
    for i in pick:
        by_frame.setdefault(int(seq[i]), []).append(i)
    meas = []
    if by_frame:
        last = max(by_frame)
        for f in frames(src, source, max_frames=max_frames, info=info):
            idx = by_frame.pop(f.seq, None)
            if idx:
                g = f.gray
                h, w = g.shape
                for i in idx:
                    c0 = work[i] + 0.5
                    o = refine.crop_origin(c0.mean(0), w, h, crop)
                    x0, y0 = int(o[0]), int(o[1])
                    for vn, bw in exposure_blur.blur_widths(g[y0:y0 + crop, x0:x0 + crop].astype(np.float64), c0 - o,
                                                            int(ids[i]), border_cells, velocity[i], sigma_optics,
                                                            vmin=vmin, dictionary=dictionary):
                        meas.append({"row": i, "marker_id": int(ids[i]), "vn": vn, "w": bw})
            if f.seq >= last or not by_frame:
                break
    est = exposure_blur.estimate_exposure_fraction(meas, min_rows=min_rows, vmin=vmin)
    est.update(n_candidates=int(len(cand)), n_sampled=len(pick), max_rows=max_rows, sample_seed=seed)
    return est


BLUR_WINDOW_RULE = ("fit loop only, per interior edge sample: w = exposure_fraction x |v . n|; half-window "
                    "h = half_std + max(0, w/2 - 1), capped at max(half_std, 2 cell - w/2 - 2 sigma_optics - 0.5) "
                    "(cell = detector side / 8), rounded up to the 0.25 px profile step; gradient centroid unchanged; "
                    "the quality pass and gate use the production window (refine.blur_half_windows)")


def detection_block(cache_npz: Path, in_prov: dict) -> dict:
    """What the refined cache needs to name the detection it was made from (embedded as `parameters.detection`):
    the detect npz's file name and sha256 (so a reader can check the detect cache beside it is the one that was
    refined), its `code` and commit, and the summary counters that describe what was detected."""
    commits = []
    for c in in_prov.get("code") or []:
        if isinstance(c, dict) and c.get("commit") and c["commit"] not in commits:
            commits.append(c["commit"])
    summary = in_prov.get("summary") or {}
    sha, sha_from = _detect_sha256(cache_npz)
    return {"file": Path(cache_npz).name, "sha256": sha, "sha256_source": sha_from,
            "code": in_prov.get("code"), "code_commits": commits,
            "summary": {k: summary.get(k) for k in ("frames_decoded", "detections")}}


def run(cache_npz: Path, *, calib, out: Path, border_cells: float | str, video: Path, crop: int = CROP,
        camera: str | None = None, max_frames: int | None = None, source: str = "nvdec_y",
        max_edge_rms_px: float = MAX_EDGE_RMS_PX, max_shift_frac: float = MAX_SHIFT_FRAC,
        min_shift_px: float = MIN_SHIFT_PX, max_downweighted_frac: float = MAX_DOWNWEIGHTED_FRAC,
        blur_window: bool = False, exposure_fraction: float | None = None, sigma_optics: float | None = None,
        min_exposure_rows: int = exposure_blur.MIN_ROWS,
        gate_agree_px: float = GATE_AGREE_PX, gate_window_frames: float = GATE_WINDOW_FRAMES,
        gate_mode: str = "resolve", samples_per_edge: int = 3, iterations: int = 3, cauchy_c: float = 1.5,
        dictionary: str | int | None = None) -> Path:
    """Refine every detection of `cache_npz` against the imagery of `video` with the pinhole calibration `calib`, and
    write the refined cache to `out`.

    `det_sharpness` is carried over from the input cache.

    Pixel convention: `refine.refine` works in the pixel-EDGE convention (pixel j spans [j, j+1)),
    every cache (detector, calibration) in the OpenCV pixel-centre convention. Seeds are
    shifted by +0.5 on the way in and refined corners by -0.5 on the way out, and the output
    provenance records `corner_convention: "centre"`, which `geometry.pose.corner_handling` requires.

    Quality gate: a converged refinement is USED only if `refine.Refinement.edge_rms_px <= max_edge_rms_px`,
    its mean corner shift `<= max(min_shift_px, max_shift_frac x detector diagonal)` and its
    `downweighted_frac <= max_downweighted_frac` (all in px of the image the refinement ran on). A gated row has
    `det_refine_used` False and `det_refine_gate` = the failed `GATE_BITS`, and is then resolved (`resolve_gated`,
    module docstring). `det_corners` holds the detector corners
    (`det_gate_resolution` 2) or the refused refinement (3), whichever agrees better with the neighbouring accepted
    rows of the same id, if within `gate_agree_px` (max corner distance, px of the image refined) of the position
    interpolated between accepted rows on both sides within `gate_window_frames` nominal frames. If both are
    farther, the row is inadmissible and dropped (arrays `inadmissible_*` keep it). With an accepted neighbour on
    one side only, or none, it is unverified (`det_gate_resolution` 4): kept with the refused refinement's corners
    (`det_unverified_corners` 1, or the detector's, 2, only if the refinement were not finite) and left to the
    consumer. The refinement is always kept in `det_corners_refined`. Non-converged rows are dropped too. Per-row
    quality: `det_refine_edge_rms_px`, `det_refine_downweighted_frac` and `det_refine_n_used` (the number of edge samples the fit used).
    `det_refine_rms_px` is the ideal-plane IRLS residual, units 1/f (see `refine.refine`).

    `gate_mode` "drop" applies the gate without the resolution and removes every gated row from
    the per-row arrays: the cache holds accepted refinements only, and carries no
    `det_gate_resolution`, `det_unverified_corners` or `det_gate_agree_px`. The removed rows are counted in
    `summary.n_gate_dropped` (they are not "inadmissible" and get no side arrays). The mode is recorded in
    `summary.gate.mode` and `parameters.gate_mode`.

    Schema note: `summary.median_rms_px` and `summary.rms_median` are aliases of `median_edge_rms_px` (edge
    residual, px), not the ideal-plane number (`median_ideal_rms_1_over_f`, units 1/f). `summary.n_used` counts
    rows whose refinement is used (det_refine_used), while the per-row `det_refine_n_used` is an edge-sample
    count.

    `parameters.detection` names the detect cache this file was refined from (`detection_block`): file, sha256,
    code commit and the detection counters.

    Border constant (`resolve_border_cells`): `border_cells` is the configuration's `marker.border_cells`
    (docs/CALIBRATION.md, "Border constant"). `parameters` and `summary` record `border_cells` and
    `border_source`.

    Blur window (opt-in, `blur_window`, `refine.refine(velocity=...)`): each row's velocity is the central
    difference of the detector quad centres, from the nearest
    earlier / later detection of the same id in time, within `MAX_NEIGHBOUR_GAP` nominal intervals (`det_pts`, so
    dropped frames do not bias it). It is in px per nominal frame (`nominal_frame_period` of `frame_pts_all`),
    one-sided at a track end, and absent for an isolated row, which is then refined without the blur window
    (`detector_velocity`). `sigma_optics` (px, the camera's at-rest optics blur) is required. There is no camera
    default. The exposure fraction is `exposure_fraction` when given (a per-camera config value), else estimated
    from this video's fast rows (`video_exposure_estimate`, refused before anything is written when fewer than
    `min_exposure_rows` fast rows can be measured). The video estimate is made in both cases and recorded beside
    the config value, so a wrong config shows. `parameters.blur_window` / `summary.blur_window` record the rule,
    sigma, the exposure fraction's source and both values (the estimate with its sd and row counts) and the
    velocity counts. `det_velocity_px` / `det_velocity_kind` hold the per-row velocity. Without `blur_window`
    neither array is written, `blur_window` is `{"enabled": false}`, and no row uses the blur window."""
    started = _dt.datetime.now(_dt.timezone.utc)
    if gate_mode not in GATE_MODES:
        raise ValueError(f"gate_mode {gate_mode!r}: one of {GATE_MODES}")
    if not blur_window and (exposure_fraction is not None or sigma_optics is not None):
        raise ValueError("exposure_fraction / sigma_optics apply only with the blur window (refine.blur_window)")
    if blur_window:
        if sigma_optics is None:
            raise ValueError("the blur window needs sigma_optics (refine.sigma_optics, the camera's at-rest optics "
                             "blur in px of the image refined); there is no camera default")
        if not (np.isfinite(sigma_optics) and sigma_optics >= 0):
            raise ValueError(f"sigma_optics {sigma_optics} must be finite and >= 0")
        if exposure_fraction is not None and not (np.isfinite(exposure_fraction) and 0 < exposure_fraction <= 2):
            raise ValueError(f"exposure_fraction {exposure_fraction} must be in (0, 2] (exposure time / frame interval)")
    cache_npz = Path(cache_npz)
    z = detect.load_arrays(cache_npz)
    in_prov = json.loads(str(z["provenance"]))
    check_calib_for_cache(cache_npz, calib)
    if calib.fisheye:
        # `refine.refine` is pinhole throughout (`projection.undistort_newton` speaks
        # k1k2p1p2k3), so refining a fisheye capture would fit the interior edges
        # in the wrong ideal plane and report a confident improvement. Refuse.
        raise ValueError(f"{cache_npz.name}: {calib.source} is a fisheye model, and the interior-edge refiner is "
                         f"pinhole throughout")
    if not calib.has_extrinsics:
        logger.warning("{}: the calibration has no camera pose. The refinement is an image-space operation and "
                       "is unaffected, but the positions derived from it are in the camera frame", cache_npz.name)
    src = _source(video)
    camera = resolve_camera(in_prov, camera)
    b_val, b_block = resolve_border_cells(border_cells)
    logger.info("{}: border_cells {} ({})", cache_npz.name, b_val,
                "default" if b_val == refine.BORDER_CELLS else b_block["border_source"])
    kw = {"border_cells": b_val}
    # refinement parameters (the configuration's `refine` group), recorded only when they differ from the defaults,
    # so the provenance of a default run has no `refine` block
    refine_params = {}
    for k, v, d in (("per_edge", samples_per_edge, 3), ("iters", iterations, 3), ("cauchy_c", cauchy_c, 1.5)):
        if v != d:
            kw[k] = v
            refine_params[k] = v
    if dictionary is not None:
        from ..detection.detectors import DICT, dictionary_id
        did = dictionary_id(dictionary)
        if did != DICT:
            import cv2
            kw["dictionary"] = cv2.aruco.getPredefinedDictionary(did)
            refine_params["dictionary"] = str(dictionary)
    out = Path(out)

    seq = z["det_seq"].astype(np.int64)
    ids = z["det_id"].astype(np.int32)
    raw = z["det_corners"].astype(np.float64)
    work = raw
    order = np.argsort(seq, kind="stable")
    by_frame: dict[int, list[int]] = {}
    for i in order:
        by_frame.setdefault(int(seq[i]), []).append(int(i))
    logger.info("{}: {} detections over {} frames; refining from {} px crops of {}",
                cache_npz.name, len(seq), len(by_frame), crop, src.name)

    keep, ref_c, rms, shift, used, reasons = [], [], [], [], [], {}
    out_c, erms, dwf, gbits = [], [], [], []       # refiner output (even if gated), quality, gate bits
    gkw = dict(max_edge_rms_px=max_edge_rms_px, max_shift_frac=max_shift_frac, min_shift_px=min_shift_px,
               max_downweighted_frac=max_downweighted_frac)
    info = probe(src)
    if tuple(calib.image_size) != (info.width, info.height):
        raise ValueError(f"{src.name} is {info.width}x{info.height} but the calibration {calib.source} is for "
                         f"{calib.image_size[0]}x{calib.image_size[1]}: calibrate at the video's resolution")
    if "det_pts" in z.files and "frame_pts_all" in z.files and len(z["frame_pts_all"]) >= 2:
        g_t = z["det_pts"].astype(np.float64) / nominal_frame_period(z["frame_pts_all"])   # nominal frames
    else:
        g_t = seq.astype(np.float64)
    vel = vkind = None
    f_used = None
    expo = None
    if blur_window:
        if "det_pts" in z.files and "frame_pts_all" in z.files and len(z["frame_pts_all"]) >= 2:
            t_row = z["det_pts"].astype(np.float64)
            period = nominal_frame_period(z["frame_pts_all"])
            time_src = "det_pts (frame timestamps), nominal period from frame_pts_all"
        else:
            t_row, period, time_src = None, None, "seq (the cache has no frame timestamps): one frame per seq step"
        vel, vkind = detector_velocity(seq, ids, work, t=t_row, period=period)
        video = video_exposure_estimate(src=src, source=source, info=info, work=work, seq=seq,
                                        ids=ids, velocity=vel, usable=np.ones(len(seq), bool), border_cells=b_val,
                                        sigma_optics=float(sigma_optics), crop=crop, max_frames=max_frames,
                                        min_rows=min_exposure_rows, dictionary=kw.get("dictionary"))
        if exposure_fraction is not None:
            f_used, f_src = float(exposure_fraction), "config"
        elif video.get("ok"):
            f_used, f_src = float(video["value"]), "video"
        else:
            raise ValueError(f"{cache_npz.name}: no exposure fraction given (refine.exposure_fraction) and the video cannot "
                             f"supply one: {video.get('reason')} (fast rows needed: {min_exposure_rows})")
        expo = {"source": f_src, "value": f_used,
                "config": None if exposure_fraction is None else float(exposure_fraction), "video": video}
        logger.info("{}: blur window: exposure fraction {} ({}; config {}, video {} +- {} from {} fast rows), "
                    "sigma_optics {} px", cache_npz.name, f_used, f_src, expo["config"], video.get("value"),
                    video.get("sd"), video.get("n_rows"), sigma_optics)
    n_errors, first_error = 0, None
    for f in (frames(src, source, max_frames=max_frames, info=info) if by_frame else ()):
        idx = by_frame.pop(f.seq, None)
        if not idx:
            continue
        g = f.gray
        h, w = g.shape
        for i in idx:
            c0 = work[i] + 0.5                          # centre-convention cache -> refiner's edge convention
            ctr = c0.mean(0)
            o = refine.crop_origin(ctr, w, h, crop)
            x0, y0 = int(o[0]), int(o[1])
            bkw = ({"velocity": vel[i], "exposure_fraction": f_used, "sigma_optics": float(sigma_optics)}
                   if blur_window and vkind[i] > 0 else {})
            try:
                r = refine.refine(g[y0:y0 + crop, x0:x0 + crop], c0 - o, int(ids[i]),
                                  calib, origin=o, **kw, **bkw)
            except Exception as e:                      # a refiner failure on one row drops that row
                reasons[type(e).__name__] = reasons.get(type(e).__name__, 0) + 1
                n_errors += 1
                first_error = first_error or f"{type(e).__name__}: {e}"
                continue
            if r is None or not r.converged:
                k = (r.reason if r is not None and r.reason else "not_converged")
                reasons[k] = reasons.get(k, 0) + 1
                continue
            q = work[i]
            diag = 0.5 * (float(np.linalg.norm(q[0] - q[2])) + float(np.linalg.norm(q[1] - q[3])))
            bits = gate(r.edge_rms_px, r.shift_px, diag, r.downweighted_frac, **gkw)
            keep.append(i)
            out_c.append(r.corners + o - 0.5)           # full-frame, back to the centre convention
            ref_c.append(out_c[-1] if bits == 0 else work[i])    # gated: the detector's corners, untouched
            rms.append(r.rms_px); shift.append(r.shift_px); used.append(r.n_used)
            erms.append(r.edge_rms_px); dwf.append(r.downweighted_frac); gbits.append(bits)
        if not by_frame:
            break
    n_tried = len(seq)
    if n_errors and n_errors * 2 > n_tried:
        # more than half the rows raised: a broken environment or input, not difficult imagery
        raise RuntimeError(f"{cache_npz.name}: the refinement raised an error on {n_errors} of {n_tried} detections, "
                           f"the first: {first_error}")
    if n_errors:
        logger.warning("{}: the refinement raised an error on {} of {} detections (dropped), the first: {}",
                       cache_npz.name, n_errors, n_tried, first_error)
    if by_frame:
        logger.warning("{}: {} frames never appeared in the decode (detections dropped: {})",
                       cache_npz.name, len(by_frame), sum(len(v) for v in by_frame.values()))
        for v in by_frame.values():
            reasons["frame_not_decoded"] = reasons.get("frame_not_decoded", 0) + len(v)

    keep = np.asarray(keep, np.int64)
    ref_c = np.asarray(ref_c, np.float64).reshape(-1, 4, 2)
    ordr = np.lexsort((ids[keep], seq[keep])) if len(keep) else np.array([], int)
    keep, ref_c = keep[ordr], ref_c[ordr]
    out_c = np.asarray(out_c, np.float64).reshape(-1, 4, 2)[ordr]
    gbits = np.asarray(gbits, np.uint8)[ordr]
    rms, shift, used = (np.asarray(v, np.float32)[ordr] for v in (rms, shift, used))
    used = used.astype(np.int32)
    erms, dwf = (np.asarray(v, np.float32)[ordr] for v in (erms, dwf))
    n_refined, n_dropped = len(keep), len(seq) - len(keep)
    # gate-failed rows: which corners (if any) may be used, see `resolve_gated`
    if gate_mode == "drop":
        # no resolution: accepted rows 1, gated rows 2 (removed below)
        gres = np.where(gbits == 0, 1, 2).astype(np.int8)
        gagree = np.full(len(keep), np.nan)
    else:
        gres, gagree = resolve_gated(g_t[keep], ids[keep], work[keep], out_c, gbits == 0, gbits != 0,
                                     agree_px=gate_agree_px, window=gate_window_frames)
    gres_counts = ({GATE_RESOLUTION[c]: int((gres == c).sum()) for c in GATE_RESOLUTION} if gate_mode == "resolve"
                   else None)
    # the refused refinement agrees better (3), or the row is unverified (4) and the refinement converged (it always
    # has, non-converged rows are dropped above): its corners. The detector's only for an unverified row without
    # finite refined corners. `det_unverified_corners` records it (0 not unverified, 1 refinement, 2 detector).
    unv = gres == UNVERIFIED
    unv_det = unv & ~np.isfinite(out_c).all(axis=(1, 2))
    unv_corners = np.where(unv, np.where(unv_det, 2, 1), 0).astype(np.int8)
    use_ref = (gres == 3) | (unv & ~unv_det)
    ref_c[use_ref] = out_c[use_ref]
    refined_used = gres == 1
    if gate_mode == "resolve":
        adm = gres != INADMISSIBLE
    else:                                               # drop: accepted refinements only, every gated row goes
        adm = gbits == 0
    dropped_gbits = gbits[~adm] if gate_mode == "drop" else np.zeros(0, np.uint8)
    if gate_mode == "drop" and len(dropped_gbits):
        logger.info("{}: gate_mode drop: {} gated rows removed (accepted refinements only)", cache_npz.name,
                    len(dropped_gbits))
    inad = {}
    if gate_mode == "resolve" and not adm.all():
        bad = ~adm
        inad = {"inadmissible_seq": seq[keep][bad], "inadmissible_id": ids[keep][bad],
                "inadmissible_pts": z["det_pts"][keep][bad], "inadmissible_gate": gbits[bad],
                "inadmissible_agree_px": gagree[bad].astype(np.float32),
                "inadmissible_corners_raw": raw[keep][bad].astype(np.float32),
                "inadmissible_corners_refined": out_c[bad].astype(np.float32)}
        logger.warning("{}: {} gate-failed rows agree with neither corner set of their two-sided "
                       "prediction and are dropped as inadmissible", cache_npz.name, int(bad.sum()))
    # drop the inadmissible rows from every per-row array, so no consumer can pose them
    keep, ref_c, out_c, gbits, rms, shift, used, erms, dwf, gres, gagree, refined_used, unv_corners = (
        v[adm] for v in (keep, ref_c, out_c, gbits, rms, shift, used, erms, dwf, gres, gagree, refined_used,
                         unv_corners))
    gated = gbits != 0
    arrays = {"det_seq": seq[keep], "det_pts": z["det_pts"][keep], "det_id": ids[keep],
              "det_corners": ref_c.astype(np.float32), "det_corners_raw": raw[keep].astype(np.float32),
              "det_corners_refined": out_c.astype(np.float32),
              "det_refine_rms_px": rms, "det_refine_shift_px": shift,
              "det_refine_edge_rms_px": erms, "det_refine_downweighted_frac": dwf,
              "det_refine_n_used": used,
              "det_refine_used": refined_used,
              "det_refine_gate": gbits,
              "det_gate_resolution": gres, "det_unverified_corners": unv_corners, "det_gate_agree_px": gagree.astype(np.float32),
              **inad,
              "frame_pts_all": z["frame_pts_all"] if "frame_pts_all" in z.files else np.array([]),
              "det_sharpness": (z["det_sharpness"][keep] if "det_sharpness" in z.files
                                else np.zeros(len(keep))),
              }
    if gate_mode == "drop":
        for k in ("det_gate_resolution", "det_unverified_corners", "det_gate_agree_px"):
            del arrays[k]
    if blur_window:
        arrays["det_velocity_px"] = vel[keep].astype(np.float32)
        arrays["det_velocity_kind"] = vkind[keep].astype(np.int8)
        vk = vkind[keep]
        bw_block = {"enabled": True, "rule": BLUR_WINDOW_RULE, "sigma_optics_px": float(sigma_optics),
                    "exposure_fraction": expo,
                    "velocity": {"from": "detector quad centres of the nearest earlier / later detection of the same "
                                         f"id within {MAX_NEIGHBOUR_GAP} nominal frame intervals: displacement / "
                                         "time x nominal period, px per nominal frame of the image refined "
                                         "(det_velocity_px; kind 2 central, 1 one-sided, 0 none = refined without "
                                         "the blur window)",
                                 "time": time_src, "nominal_period_s": period,
                                 "n_central": int((vk == 2).sum()), "n_one_sided": int((vk == 1).sum()),
                                 "n_none": int((vk == 0).sum())}}
    else:
        bw_block = {"enabled": False}

    def _med(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return round(float(np.median(v)), 4) if len(v) else None

    def _p95(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return round(float(np.percentile(v, 95)), 4) if len(v) else None

    all_bits = np.concatenate([gbits, inad.get("inadmissible_gate", np.zeros(0, np.uint8)), dropped_gbits])
    reasons_g = {k: int(((all_bits & b) != 0).sum()) for k, b in GATE_BITS.items()}
    summary = {"source_cache": cache_npz.name, "video": src.name, "crop_px": crop,
               "n_in": int(len(seq)), "n_refined": n_refined,
               "n_dropped": n_dropped, "drop_reasons": reasons,
               "convergence_rate": round((len(keep) + len(dropped_gbits)) / max(1, len(seq)), 5),
               "n_used": int(refined_used.sum()), "n_gated": int(gated.sum()) + len(inad.get("inadmissible_id", ())) + len(dropped_gbits),
               "n_gate_dropped": len(dropped_gbits),
               "gate_reasons": reasons_g,
               "n_inadmissible": len(inad.get("inadmissible_id", ())), "gate_resolution": gres_counts,
               "gate": ({**gkw, "bits": GATE_BITS, "far_px": refine.FAR_PX, "mode": "resolve", "agree_px": gate_agree_px,
                         "window_frames": gate_window_frames} if gate_mode == "resolve"
                        else {**gkw, "bits": GATE_BITS, "far_px": refine.FAR_PX, "mode": gate_mode}),
               "median_shift_px": _med(shift),
               "median_edge_rms_px": _med(erms), "p95_edge_rms_px": _p95(erms),
               "median_ideal_rms_1_over_f": _med(rms),
               **b_block,
               "corner_convention": (DROP_CORNER_CONVENTION if gate_mode == "drop" else
                                     "pixel centre; REFINED where det_refine_used, on gated rows (det_refine_gate != 0) "
                                     "the corners det_gate_resolution names (detector or refused refinement, "
                                     "whichever agrees with the accepted neighbours; unverified rows, with a neighbour on one side "
                                     "only or none, the refused refinement; inadmissible rows are dropped); raw "
                                     "detector corners in det_corners_raw, the refiner's output in "
                                     "det_corners_refined"),
               "source": source, "blur_window": bw_block}
    # `rms_median`/`median_rms_px` are the edge residual in px. The 1/f number is median_ideal_rms_1_over_f.
    summary["median_rms_px"] = summary["median_edge_rms_px"]
    summary["rms_median"] = summary["median_edge_rms_px"]
    prov = provenance.block(
        relpath=in_prov["input"]["relpath"], decoder=in_prov["environment"]["decoder"],
        pixel_pipeline=in_prov["environment"]["pixel_pipeline"],
        params={"refined_from": cache_npz.name, "detector": in_prov["parameters"],
                "detection": detection_block(cache_npz, in_prov),
                "crop_px": crop, **b_block,
                "calibration": {"source": calib.source, "sha256": calib.sha256,
                                "has_extrinsics": calib.has_extrinsics},
                "blur_window": bw_block, "gate_mode": gate_mode,
                "refine_decoder": decoder_label(source, info.codec),
                **({"refine": refine_params} if refine_params else {})},
        started=started, extra={"summary": summary, "corner_convention": "centre", "camera": camera or None})
    # atomic, like the detect cache: a killed refine never leaves a truncated .refined.npz behind
    detect._write_cache(Path(out), {"provenance": np.array(json.dumps(prov)), **arrays}, prov)
    logger.info("{}: refined {}/{} ({:.3f}), {} gated {}, resolution {}, median shift {} px, edge rms {} px -> {}",
                cache_npz.name, summary["n_refined"], summary["n_in"], summary["convergence_rate"],
                summary["n_gated"], reasons_g, gres_counts, summary["median_shift_px"], summary["median_edge_rms_px"],
                out.name)
    if reasons:
        logger.info("  dropped: {}", reasons)
    return out


def resolve_camera(prov: dict, override: str | None = None) -> str:
    """The camera name of a cache: `override` when given (configuration `camera.name`), else the
    `camera` its provenance records, else "" (unnamed)."""
    return override or prov.get("camera") or ""
