"""Single-camera trajectories: refined cache -> per-detection marker pose -> floor -> gates -> smooth / interp ->
files (the paper, Section 3.6, "Pose, trajectories and output").

Calibration, walking surfaces, dataset and camera names come in from the caller. Nothing is looked up. Steps, in
this order (reordering them changes which rows are kept):

1. `measured_rows`: one IPPE marker pose per kept detection (`geometry.pose.derive`, the solution with the lower
   reprojection error), the world-z gate (`track.z_range_m`), then the first row per (id, frame). The gate runs
   before the de-duplication, so a duplicate outside the gate cannot displace a valid row. `row_rule` says which
   rows of the cache are kept: with gate resolutions, codes 1, 2 and 3 (refined, or detector corners or refused
   refinement confirmed by both accepted neighbours), not 4 (unverified). Without them (gate mode drop), the
   `det_refine_used` rows, which are all of its rows.
2. `attach_floor`: the marker centre dropped vertically onto the walking surfaces (`floor.model: surfaces`), or onto
   the plane z = 0 (`floor.model: plane`, which `flat_floor` builds as one surface, `FLAT` being its name here).
3. `height_gate`: rows with a surface whose marker height above it is outside `track.height_band_m` are dropped.
4. `orient_arrays` (marker orientation for a viewer), `smooth_cols` (point smoother per segment with the
   acceleration prior `track.sigma_acc_m` and the measurement noise `track.sigma_s`, estimated from second
   differences unless fixed), floor and height gate again on the smoothed rows, and `interp_cols` (linear fill of
   gaps up to `track.interp_cap_frames`).
5. Files: `<stem>.fused.txt`, `.fused_interp.txt`, `.fused_smooth.txt` (`trajwrite`), `.petrack.txt` (with its
   `observed` sidecar), `.orient.npz` and `.sidecar.json`. A stamp in the work folder holds a hash of every input,
   and a run with the same hash writes nothing.

Columns of a track set (`cols`): `trajwrite.COLUMNS`. Source 1 = measured by the camera, 0 = interpolated."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import provenance
from .config import SIGMA_S_MAX_M, sigma_s_given
from .geometry import marker_orient as OR
from .geometry import surfaces as S
from .io import trajwrite as TW
from .refinement.gate import GATE_RESOLUTION

SIGMA_ACC_M = 0.00045         # acceleration prior, metres / frame^2 (the paper, Appendix A)
SIGMA_S_FALLBACK_M = 0.002
SIGMA_S_FLOOR_M = 1e-4
MIN_TRIPLES = 20
KEEP_RESOLUTIONS = (1, 2, 3)
FLAT = "plane"                # `floor: plane`: one synthetic surface z = 0 under everything
FLAT_EXTENT_M = 1.0e4


@dataclass(frozen=True)
class Params:
    """Everything that changes a number in the output (part of the key that lets an unchanged re-run skip the
    trajectory step). Built from the run configuration's
    `track` group (`from_config`). None turns a gate off (a calibration without extrinsics has no world z)."""
    z_min: float | None = -3.0     # world-z gate on the head point, wide so that walking surfaces below and above
                                   # the z = 0 plane (e.g. stairs) stay inside it
    z_max: float | None = 4.5
    height_min: float | None = 0.3  # marker height above z_floor, applied only to rows that have a surface
    height_max: float | None = 2.8
    max_gap: int = 30
    min_segment: int = 5           # measured rows (shorter segments are dropped from the smoothed set)
    sigma_acc_m: float = SIGMA_ACC_M
    sigma_s_m: float | None = None  # measurement noise of the smoother (each axis), None = estimated per clip (`sigma_s`)
    interp_cap: int = 30
    border_cells: float = 0.869    # the refinement's border constant (recorded here, the refinement applies it)

    def __post_init__(self):
        if self.sigma_s_m is not None and not 0.0 < float(self.sigma_s_m) <= SIGMA_S_MAX_M:
            raise ValueError(f"sigma_s_m={self.sigma_s_m!r}: None (estimated) or in (0, {SIGMA_S_MAX_M}] m")

    @classmethod
    def from_config(cls, cfg) -> "Params":
        """The `track` group (plus `marker.border_cells`) of an `arucotrack.config` configuration."""
        t = cfg.track
        z = list(t.z_range_m) if t.z_range_m is not None else [None, None]
        h = list(t.height_band_m) if t.height_band_m is not None else [None, None]
        return cls(z_min=z[0], z_max=z[1], height_min=h[0], height_max=h[1], max_gap=int(t.max_gap_frames),
                   min_segment=int(t.min_segment_rows), sigma_acc_m=float(t.sigma_acc_m),
                   sigma_s_m=sigma_s_given(t.sigma_s), interp_cap=int(t.interp_cap_frames), border_cells=float(cfg.marker.border_cells))

    def text(self, marker_m: float) -> str:
        zg = "off" if self.z_min is None else f"{self.z_min}..{self.z_max} m"
        hg = "off" if self.height_min is None else f"{self.height_min}..{self.height_max} m"
        return (f"world z gate {zg} | head height above z_floor {hg} "
                f"(rows with a surface) | max gap {self.max_gap} frames | min segment {self.min_segment} measured rows | "
                f"sigma_acc {self.sigma_acc_m * 1000:g} mm/frame^2 | "
                + ("" if self.sigma_s_m is None else f"sigma_s {self.sigma_s_m * 1000:g} mm (fixed) | ")
                + f"interpolation cap {self.interp_cap} frames | "
                f"refine border_cells {self.border_cells:.6f} | marker side {marker_m:.4f} m")


def empty_cols() -> dict:
    c = {k: np.zeros(0) for k in TW.FLOAT_COLS}
    for k in TW.INT_COLS:
        c[k] = np.zeros(0, int)
    for k in TW.STR_COLS:
        c[k] = np.zeros(0, object)
    return c


def concat(parts: list) -> dict:
    parts = [p for p in parts if len(p["id"])]
    if not parts:
        return empty_cols()
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in TW.COLUMNS}


def take(c: dict, idx) -> dict:
    return {k: np.asarray(c[k])[idx] for k in TW.COLUMNS}


# ---------------------------------------------------------------- measured rows

def offnadir_deg(calib, corners4) -> float:
    """Angle (deg) of the marker-centre ray: from world -z (the off-nadir angle) with a camera pose, from the optical
    axis +z without one (camera-frame output). The centre is the diagonals' intersection in normalised
    coordinates."""
    xn = np.asarray(calib.normalise(np.asarray(corners4, float).reshape(4, 2)), float).reshape(4, 2)
    if not np.isfinite(xn).all():
        return float("nan")
    p, r = xn[0], xn[2] - xn[0]
    q, s = xn[1], xn[3] - xn[1]
    den = r[0] * s[1] - r[1] * s[0]
    c = xn.mean(0) if abs(den) < 1e-14 else p + r * (((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den)
    if not calib.has_extrinsics:
        return float(np.degrees(np.arccos(1.0 / np.hypot(np.hypot(c[0], c[1]), 1.0))))
    d = calib.Rt @ np.array([c[0], c[1], 1.0])
    return float(np.degrees(np.arccos(np.clip(-d[2] / np.linalg.norm(d), -1, 1))))


def row_rule(z) -> tuple[np.ndarray, str]:
    """(keep mask, rule text) for the rows of a refined cache (an open npz or a dict of its arrays)."""
    files = z.files if hasattr(z, "files") else list(z)
    if "det_gate_resolution" in files:
        res = np.asarray(z["det_gate_resolution"]).astype(int)
        return (np.isin(res, KEEP_RESOLUTIONS),
                f"gate resolutions {', '.join(map(str, KEEP_RESOLUTIONS))} kept (refined; gate-failed rows checked "
                f"against both accepted neighbours), 4 unverified dropped")
    return np.asarray(z["det_refine_used"]).astype(bool), "det_refine_used rows (accepted refinements) only"


def measured_rows(cache, geom, calib, params: Params) -> tuple[dict, dict]:
    """One IPPE marker pose per kept detection (`row_rule`), first row per (id, frame), world-z gated before the
    de-duplication. Returns (cols, info)."""
    from .geometry.pose import radius
    with np.load(cache, allow_pickle=False) as z:
        seq = z["det_seq"].astype(np.int64); ids = z["det_id"].astype(int)
        cor = z["det_corners"].astype(np.float64)
        used, rule = row_rule(z)
        pts = np.asarray(z["frame_pts_all"], float)
        # which corners each row was posed from: the gate resolution where the cache has one, else accepted
        # refinements only (`row_rule` keeps no other rows then)
        corner_src = (np.array([GATE_RESOLUTION[int(k)] for k in z["det_gate_resolution"]], object)
                      if "det_gate_resolution" in z.files else np.full(len(seq), "refined", object))
    with np.load(geom, allow_pickle=False) as g:
        # world `det_xyz` with extrinsics, camera-frame `det_xyz_cam` without (geometry.pose.derive names the frame)
        xyz = g["det_xyz" if calib.has_extrinsics else "det_xyz_cam"]
        res, gseq = g["det_residual"], g["det_seq"]
    if len(gseq) != len(seq) or not (gseq == seq).all():
        raise ValueError(f"{Path(geom).name} does not match {Path(cache).name} row by row")
    r = radius(calib, cor.mean(1))
    info = {"n_rows": int(len(seq)), "dropped_unrefined": int((~used).sum()), "row_rule": rule,
            "z_gated": {}, "duplicates": {}, "too_few_rows": {}, "row": {}}
    parts = []
    for mid in np.unique(ids[used]):
        sel = np.flatnonzero((ids == mid) & used & np.isfinite(xyz).all(axis=1))
        sel = sel[np.argsort(seq[sel], kind="stable")]
        ok = (np.ones(len(sel), bool) if params.z_min is None
              else (xyz[sel, 2] > params.z_min) & (xyz[sel, 2] < params.z_max))     # gate before the dedupe
        if (~ok).any():
            info["z_gated"][str(int(mid))] = int((~ok).sum())
        sel = sel[ok]
        _, first = np.unique(seq[sel], return_index=True)
        if len(first) < len(sel):
            info["duplicates"][str(int(mid))] = int(len(sel) - len(first))
        sel = sel[first]
        if len(sel) < 2:                                # one row is no track
            if len(sel):
                info["too_few_rows"][str(int(mid))] = int(len(sel))
            continue
        info["row"].update({(int(mid), int(seq[k])): int(k) for k in sel})
        n = len(sel)
        c = empty_cols()
        c.update(id=np.full(n, mid), frame=seq[sel], t_s=pts[seq[sel]], x=xyz[sel, 0], y=xyz[sel, 1], z=xyz[sel, 2],
                 observed=np.ones(n, int), source=np.ones(n, int), miss_mm=np.full(n, np.nan),
                 ray_angle_deg=np.full(n, np.nan), view_deg=np.array([offnadir_deg(calib, cor[k]) for k in sel]),
                 corner=corner_src[sel], r_norm=r[sel], reproj_px=res[sel],
                 flags=np.array(["-"] * n, object), x_floor=np.full(n, np.nan), y_floor=np.full(n, np.nan),
                 z_floor=np.full(n, np.nan), surface_id=np.full(n, -1), floor_src=np.zeros(n, int))
        parts.append(c)
    return concat(parts), {**info, "pts": pts}


# ---------------------------------------------------------------- orientation

def orient_arrays(cols: dict, rows: dict, cache, calib, half_s: float = 0.1) -> dict:
    """<stem>.orient.npz arrays for the measured rows `cols` (id, frame, t_s used): world-from-marker quaternion (xyzw, IPPE
    flips rejected against the smoothed orientation, then window-averaged +-half_s), the rejection flag and the raw refined
    image corners (px) for the video overlay."""
    with np.load(cache, allow_pickle=False) as z:
        cor_all = z["det_corners"].astype(np.float64)
    ids, fr, t = np.asarray(cols["id"]), np.asarray(cols["frame"]), np.asarray(cols["t_s"])
    out = {k: [] for k in ("id", "frame", "quat_xyzw", "ippe_flip_rejected", "corners_px")}
    for mid in np.unique(ids):
        k = np.flatnonzero(ids == mid)
        k = k[np.argsort(fr[k], kind="stable")]
        ridx = np.array([rows[(int(mid), int(fr[j]))] for j in k])
        cor = cor_all[ridx]
        cand, _ = OR.candidates(calib, cor)
        good = np.isfinite(cand).all(axis=(1, 2))
        if not good.any():
            continue
        k, cor, cand = k[good], cor[good], cand[good]
        pick, sm, rej = OR.choose_and_smooth(cand, t[k], half_s)
        out["id"].append(ids[k]); out["frame"].append(fr[k]); out["quat_xyzw"].append(sm)
        out["ippe_flip_rejected"].append(rej); out["corners_px"].append(cor)
    if not out["id"]:
        return {"id": np.zeros(0, int), "frame": np.zeros(0, int), "quat_xyzw": np.zeros((0, 4)),
                "ippe_flip_rejected": np.zeros(0, bool), "corners_px": np.zeros((0, 4, 2))}
    return {"id": np.concatenate(out["id"]).astype(np.int32), "frame": np.concatenate(out["frame"]).astype(np.int32),
            "quat_xyzw": np.concatenate(out["quat_xyzw"]), "ippe_flip_rejected": np.concatenate(out["ippe_flip_rejected"]),
            "corners_px": np.concatenate(out["corners_px"]).astype(np.float32)}


# ---------------------------------------------------------------- floor

def flat_floor(extent_m: float = FLAT_EXTENT_M) -> list:
    """`floor: plane`: one surface, the plane z = 0 (id 0), covering every x, y within +-extent_m."""
    e = float(extent_m)
    return [S.Surface(0, "plane", np.array([[-e, -e], [e, -e], [e, e], [-e, e]], float), (0.0, 0.0, 1.0, 0.0))]


def extrapolated_ids(surfaces_json) -> set:
    """Ids of the surfaces the surfaces document marks `source: "extrapolated"` (continued, not fitted to the mesh)."""
    doc = json.loads(Path(surfaces_json).read_text())
    return {int(s["id"]) for s in doc["surfaces"] if s.get("source") == "extrapolated"}


def attach_floor(cols: dict, surfaces, extrapolated_ids=frozenset()) -> dict:
    """x_floor/y_floor/z_floor/surface_id from `project_to_surfaces`, NaN / -1 when there are no surfaces (None).
    floor_src: 0 no surface, 1 a measured surface, 2 a surface in `extrapolated_ids`. In place."""
    n = len(cols["id"])
    cols["x_floor"] = np.full(n, np.nan); cols["y_floor"] = np.full(n, np.nan)
    cols["z_floor"] = np.full(n, np.nan); cols["surface_id"] = np.full(n, -1); cols["floor_src"] = np.zeros(n, int)
    if surfaces is None or n == 0:
        return cols
    xf, sid = S.project_to_surfaces(np.c_[cols["x"], cols["y"], cols["z"]], surfaces)
    cols["x_floor"], cols["y_floor"], cols["z_floor"] = xf[:, 0], xf[:, 1], xf[:, 2]
    sid = np.asarray(sid, int)
    cols["surface_id"] = sid
    ext = np.isin(sid, np.fromiter(extrapolated_ids, int, len(extrapolated_ids)))
    cols["floor_src"] = np.where(sid < 0, 0, np.where(ext, 2, 1))
    return cols


def height_gate(cols: dict, params: Params, has_surfaces: bool) -> tuple[dict, int]:
    """Drop rows that have a surface and whose marker height above z_floor is outside the band. Rows without a surface stay."""
    if not has_surfaces or not len(cols["id"]) or params.height_min is None:
        return cols, 0
    h = np.asarray(cols["z"]) - np.asarray(cols["z_floor"])
    bad = (np.asarray(cols["surface_id"]) >= 0) & ((h < params.height_min) | (h > params.height_max))
    return take(cols, np.flatnonzero(~bad)), int(bad.sum())


# ---------------------------------------------------------------- smoothing

def sigma_s(cols: dict) -> dict:
    """1.4826 MAD / sqrt(6) of second differences over runs of three consecutive frames, pooled over ids and x, y, z.
    Fewer than MIN_TRIPLES triples give the 2 mm fallback. The result is at least 0.1 mm."""
    vals = []
    ids, fr = np.asarray(cols["id"]), np.asarray(cols["frame"])
    P = np.c_[cols["x"], cols["y"], cols["z"]]
    for mid in np.unique(ids):
        k = np.flatnonzero(ids == mid)
        k = k[np.argsort(fr[k], kind="stable")]
        for a, b, c in zip(k[:-2], k[1:-1], k[2:]):
            if fr[c] - fr[a] == 2:
                vals.append(P[c] - 2 * P[b] + P[a])
    if len(vals) >= MIN_TRIPLES:
        v = np.concatenate(vals)
        sig, origin = 1.4826 * float(np.median(np.abs(v - np.median(v)))) / np.sqrt(6.0), "source"
    else:
        sig, origin = SIGMA_S_FALLBACK_M, "fallback 2 mm"
    return {"1": {"sigma_m": max(sig, SIGMA_S_FLOOR_M), "origin": origin, "n_triples": len(vals)}}


def smooth_segment(frames, y, sig, sigma_acc_m: float = SIGMA_ACC_M):
    """min sum ||p_k - y_k||^2/sigma_k^2 + sum ||p_{k+1} - 2 p_k + p_{k-1}||^2/sigma_acc^2 over every frame of the segment."""
    frames = np.asarray(frames, int)
    F = np.arange(frames[0], frames[-1] + 1)
    n = len(F)
    idx = frames - frames[0]
    w = np.zeros(n); w[idx] = 1.0 / np.asarray(sig, float) ** 2
    rhs = np.zeros((n, 3)); rhs[idx] = np.asarray(y, float) * w[idx][:, None]
    A = sparse.diags(w)
    if n >= 3:
        D = sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
        A = A + (D.T @ D) / sigma_acc_m ** 2
    return F, np.asarray(spsolve(sparse.csc_matrix(A), rhs), float).reshape(n, 3)


def segments(frames, max_gap: int, min_rows: int) -> list:
    """(i0, i1, kept) over sorted rows, split at gaps > max_gap, kept when it has >= min_rows measured rows."""
    frames = np.asarray(frames, int)
    if not len(frames):
        return []
    brk = np.flatnonzero(np.diff(frames) > max_gap)
    starts = np.r_[0, brk + 1]; ends = np.r_[brk + 1, len(frames)]
    return [(int(a), int(b), bool(b - a >= min_rows)) for a, b in zip(starts, ends)]


def smooth_cols(cols: dict, pts, params: Params, sig: dict | None = None) -> tuple[dict, dict]:
    """fused_smooth: per id and segment the point smoother. Filled frames are source 0 / observed 0 with every
    measurement column NaN / '-'. Measured frames carry their raw row's columns. Returns (cols, info{dropped_segments}).
    `params.sigma_s_m` set: that measurement noise replaces the estimate (origin "parameter", n_triples still counted)."""
    sig = sig or sigma_s(cols)
    if params.sigma_s_m is not None:
        sig = {"1": {"sigma_m": float(params.sigma_s_m), "origin": "parameter", "n_triples": sig["1"]["n_triples"]}}
    s = sig["1"]["sigma_m"]
    out, dropped = [], {}
    for mid in np.unique(cols["id"]):
        k = np.flatnonzero(cols["id"] == mid)
        k = k[np.argsort(np.asarray(cols["frame"])[k], kind="stable")]
        c = take(cols, k)
        for a, b, kept in segments(c["frame"], params.max_gap, params.min_segment):
            if not kept:
                dropped.setdefault(str(int(mid)), []).append({"first": int(c["frame"][a]), "last": int(c["frame"][b - 1]),
                                                              "n_rows": int(b - a)})
                continue
            fr = np.asarray(c["frame"][a:b])
            F, P = smooth_segment(fr, np.c_[c["x"][a:b], c["y"][a:b], c["z"][a:b]], np.full(b - a, s), params.sigma_acc_m)
            n = len(F)
            seg = empty_cols()
            seg.update(id=np.full(n, mid), frame=F, t_s=np.asarray(pts, float)[F], x=P[:, 0], y=P[:, 1], z=P[:, 2],
                       observed=np.zeros(n, int), source=np.zeros(n, int),
                       corner=np.array(["-"] * n, object), flags=np.array(["-"] * n, object),
                       surface_id=np.full(n, -1), floor_src=np.zeros(n, int))
            for col in TW.FLOAT_COLS:
                if col not in ("t_s", "x", "y", "z"):
                    seg[col] = np.full(n, np.nan)
            at = (fr - F[0])
            for col in ("observed", "source", "miss_mm", "ray_angle_deg", "view_deg", "corner", "r_norm",
                        "reproj_px", "flags"):
                seg[col][at] = np.asarray(c[col][a:b])
            out.append(seg)
    return concat(out), {"dropped_segments": dropped, "sigma_s": sig}


def interp_cols(cols: dict, pts, cap: int) -> dict:
    """fused_interp: raw rows plus linear fill of gaps <= cap frames (observed 0, source 0, measurement columns NaN)."""
    pts = np.asarray(pts, float)
    out = []
    for mid in np.unique(cols["id"]):
        k = np.flatnonzero(cols["id"] == mid)
        k = k[np.argsort(np.asarray(cols["frame"])[k], kind="stable")]
        c = take(cols, k)
        fr = np.asarray(c["frame"])
        new = []
        for i in range(len(fr) - 1):
            gap = int(fr[i + 1] - fr[i])
            if 1 < gap <= cap:
                new += [(i, g, g / gap) for g in range(1, gap)]
        if not new:
            out.append(c)
            continue
        n = len(new)
        f = empty_cols()
        i0 = np.array([t[0] for t in new]); w = np.array([t[2] for t in new])
        f.update(id=np.full(n, mid), frame=fr[i0] + np.array([t[1] for t in new]), observed=np.zeros(n, int),
                 source=np.zeros(n, int), corner=np.array(["-"] * n, object), flags=np.array(["interp"] * n, object),
                 surface_id=np.full(n, -1), floor_src=np.zeros(n, int))
        for col in TW.FLOAT_COLS:
            f[col] = np.full(n, np.nan)
        for col in ("x", "y", "z"):
            f[col] = np.asarray(c[col])[i0] * (1 - w) + np.asarray(c[col])[i0 + 1] * w
        f["t_s"] = pts[f["frame"]]
        out += [c, f]
    r = concat(out)
    return take(r, np.lexsort((r["frame"], r["id"]))) if len(r["id"]) else r


def framerate_from_pts(pts) -> float:
    """(n_packets - 1) / (PTS_last - PTS_first): the packet rate, which counts dropped packets as absent. Needs at
    least two packets with distinct timestamps (ValueError otherwise)."""
    pts = np.asarray(pts, float)
    if len(pts) < 2 or not pts[-1] > pts[0]:
        raise ValueError(f"the frame rate needs at least two frames with distinct timestamps ({len(pts)} given)")
    return float((len(pts) - 1) / (pts[-1] - pts[0]))


# ---------------------------------------------------------------- provenance

def code_state() -> dict:
    """The package version and its git commit (`-dirty` appended when a tracked file of the checkout has uncommitted
    changes), or "unknown" when the package is not running from its own checkout (`provenance.code_state`). The
    commit is part of the work-dir key only, never of a file in `out/`."""
    import arucotrack

    st = provenance.code_state()
    describe = "unknown" if st["commit"] is None else st["commit"] + ("-dirty" if st["dirty"] else "")
    return {"arucotrack": {"version": arucotrack.__version__, "git_describe": describe}}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


# ---------------------------------------------------------------- the metric step

FRAME_WORLD = ("units: metres, in the world frame of the calibration's camera pose (z up, origin and axes as the "
               "pose defines them); x y z = the marker centre")
FRAME_CAMERA = ("units: metres, in the CAMERA frame (the calibration has no camera pose): origin at the camera, "
                "x right, y down, z along the optical axis; x y z = the marker centre, no floor")
FRAME_INDEX = "frame: the frame index of the video, from 0; t_s: that frame's timestamp, seconds"
VARIANT_TEXT = {"fused": "measured rows, one per frame and marker",
                "fused_interp": "measured rows, gaps up to the fill limit filled linearly",
                "fused_smooth": "smoothed per segment, every frame of a segment"}


def track_text(params: Params) -> str:
    """The header line of the track parameters."""
    gate = ("" if params.z_min is None else f"z gate {params.z_min}..{params.z_max} m | ") + (
        "" if params.height_min is None else f"height above the floor {params.height_min}..{params.height_max} m | ")
    return (f"track: {gate}max gap {params.max_gap} frames | min segment {params.min_segment} rows | "
            f"sigma_acc {params.sigma_acc_m * 1000:g} mm/frame^2 | "
            + ("" if params.sigma_s_m is None else f"sigma_s {params.sigma_s_m * 1000:g} mm | ")
            + f"linear fill up to {params.interp_cap} frames")


def header_lines(stem, calib, params, fps, floor_text, code, variant) -> list:
    """The `#` lines of a trajectory file: what the file is, its frame and units, and what made it."""
    cx = calib.cam
    version = code["arucotrack"]["version"]
    return [f"arucotrack {version} trajectories | clip {stem} | {variant}: {VARIANT_TEXT[variant]}",
            FRAME_WORLD if calib.has_extrinsics else FRAME_CAMERA,
            f"frame rate: {fps:.4f} fps",
            FRAME_INDEX,
            # file name only: a machine path does not belong in shared output files
            f"calibration: {Path(str(calib.source)).name}",
            f"camera centre: ({cx[0]:.4f}, {cx[1]:.4f}, {cx[2]:.4f}) m",
            f"marker: side {calib.marker_side:.4f} m, border constant {params.border_cells:.4f}",
            track_text(params),
            f"floor: {floor_text}",
            TW.SOURCE_TABLE, TW.OBSERVED_TABLE, TW.FLOOR_SRC_TABLE, TW.CODED_TEXT_TABLE]


def inputs_key(refined_sha, calib_sha, names, params, floor_sha, code) -> str:
    blob = json.dumps({"refined": refined_sha, "calib": calib_sha, "names": names, "params": asdict(params),
                       "floor": floor_sha, "code": code}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def resolve_floor(floor) -> tuple[list | None, str | None, set, str]:
    """(surfaces, sha or tag, extrapolated ids, header text) for `floor`: None (no floor), `FLAT` or a surfaces JSON."""
    if floor is None:
        return None, None, set(), "none (floor columns nan, surface_id -1)"
    if floor == FLAT:
        return flat_floor(), "plane z=0", set(), "the plane z = 0 under the marker centre"
    p = Path(floor)
    sha = provenance.sha256_file(p)
    return (S.load_surfaces(p), sha, extrapolated_ids(p),
            f"the walking surfaces of {p.name} under the marker centre")


def metric_step(stem: str, refined: Path, detect_cache: Path, out_dir: Path, work: Path, *, calib, floor=None,
                params: Params = Params(), camera: str = "camera", force: bool = False,
                config_line: str | None = None) -> dict:
    """refined cache -> fused / fused_interp / fused_smooth txt, petrack txt + observed.json, orient.npz, sidecar.
    `calib`: a `calibration.camera.Calib` (`Calib.from_json`). `floor`: None, `FLAT` ("plane") or a
    ground_surfaces.json path. Idempotent: the stamp `<work>/<stem>.trajectories.run.json` holds a hash of every
    input, and a re-run with the same hash, with every output present, writes nothing.
    A calibration without extrinsics gives camera-frame positions: then `floor` must be None and the z gate off
    (`params.z_min` None), ValueError otherwise. A fisheye calibration is refused (ValueError): the trajectory step
    poses with the pinhole model. `config_line` (the configuration summary) is recorded in the sidecar."""
    if calib.fisheye:
        raise ValueError(f"{calib.source}: a fisheye calibration, and the trajectory step poses with the pinhole model")
    if not calib.has_extrinsics and (floor is not None or params.z_min is not None or params.height_min is not None):
        raise ValueError(f"{calib.source}: no extrinsics, so positions are camera-frame; a floor model and the world-z "
                         f"and height gates need a world frame (floor None, z and height gates off)")
    from .geometry.pose import derive
    from .io import petrack_export as PE
    out_dir, work = Path(out_dir), Path(work)
    names = {"camera": camera, "method": Path(calib.source).name}
    surfaces, floor_sha, ext_ids, floor_text = resolve_floor(floor)
    code = code_state()
    refined_sha = provenance.sha256_file(refined)
    key = inputs_key(refined_sha, calib.sha256, names, params, floor_sha, code)
    if config_line:
        key = hashlib.sha256((key + config_line).encode()).hexdigest()
    # the output folder is part of the key: several output folders may share one work folder
    key = hashlib.sha256((key + str(out_dir.resolve())).encode()).hexdigest()
    stamp = work / f"{stem}.trajectories.run.json"
    files = {v: out_dir / f"{stem}.{v}.txt" for v in ("fused", "fused_interp", "fused_smooth")}
    pet = out_dir / f"{stem}.petrack.txt"
    side = out_dir / f"{stem}.sidecar.json"
    orient_f = out_dir / f"{stem}.orient.npz"
    outs = [*files.values(), pet, out_dir / f"{stem}.petrack.observed.json", orient_f, side]
    if not force and stamp.is_file() and all(o.is_file() for o in outs):
        if json.loads(stamp.read_text()).get("inputs_key") == key:
            return {"skipped": True, "inputs_key": key}
    work.mkdir(parents=True, exist_ok=True)
    # the outputs are rewritten from here: without the stamp an interrupted rewrite is redone on the next run
    stamp.unlink(missing_ok=True)
    # one geom file per refined cache and calibration, so runs of two gate modes in one work dir do not share it
    # the geom file is named after the calibration (its file name without the extension)
    geom = derive(refined, calib,
                  out=work / Path(refined).name.replace(".npz", f".{Path(names['method']).stem or 'calib'}.geom.npz"))
    raw, info = measured_rows(refined, geom, calib, params)
    pts = info.pop("pts")
    fps = framerate_from_pts(pts)
    attach_floor(raw, surfaces, ext_ids)
    raw, n_hgate_raw = height_gate(raw, params, surfaces is not None)
    ori = orient_arrays(raw, info.pop("row"), refined, calib)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = orient_f.with_name(orient_f.name + ".tmp")
    with open(tmp, "wb") as fh:                         # a file handle: savez would append .npz to the name
        np.savez_compressed(fh, **ori)
    os.replace(tmp, orient_f)
    smooth, sinfo = smooth_cols(raw, pts, params)
    attach_floor(smooth, surfaces, ext_ids)
    smooth, n_hgate_smooth = height_gate(smooth, params, surfaces is not None)
    interp = attach_floor(interp_cols(raw, pts, params.interp_cap), surfaces, ext_ids)
    with np.load(detect_cache, allow_pickle=False) as cs:
        det_seq = np.unique(cs["det_seq"])
        n_frames = int(len(cs["frame_pts_all"]))
        det_ids = np.asarray(cs["det_id"])
    detection = {"n_frames": n_frames, "frames_with_marker": int(len(det_seq)),
                 "rate": float(len(det_seq) / n_frames) if n_frames else None,
                 "ids": sorted(int(i) for i in np.unique(det_ids)), "n_detections": int(len(det_ids))}
    rule = info["row_rule"]
    for v, c in (("fused", raw), ("fused_interp", interp), ("fused_smooth", smooth)):
        TW.write_interface(files[v], c, header_lines(stem, calib, params, fps, floor_text, code, v))
    tracks = {int(m): np.column_stack([smooth["frame"][smooth["id"] == m], smooth["x"][smooth["id"] == m],
                                       smooth["y"][smooth["id"] == m], smooth["z"][smooth["id"] == m],
                                       smooth["observed"][smooth["id"] == m]]).astype(float)
              for m in np.unique(smooth["id"])}
    o = {m: t[np.argsort(t[:, 0])] for m, t in tracks.items()}
    PE.write(pet, o, variant="fused_smooth", fps=fps, camera_frame=not calib.has_extrinsics)
    per_id = {str(int(m)): {"n_measured": int(((raw["id"] == m)).sum()), "n_smooth": int((smooth["id"] == m).sum()),
                            "first_frame": int(raw["frame"][raw["id"] == m].min()), "last_frame": int(raw["frame"][raw["id"] == m].max())}
              for m in np.unique(raw["id"])}
    sidecar = {"stem": stem, "camera": camera,
               "frame": "world" if calib.has_extrinsics else "camera", "config": config_line,
               # input files by name: sidecars carry no machine paths
               "calibration": {"path": Path(str(calib.source)).name, "fx": float(calib.K[0, 0]),
                               "centre_m": [float(v) for v in calib.cam], "marker_m": calib.marker_side},
               "parameters": asdict(params), "parameters_text": params.text(calib.marker_side), "row_rule": rule,
               "code": f"arucotrack {code['arucotrack']['version']}",
               "detection_cache": Path(detect_cache).name, "refined_cache": Path(refined).name,
               "floor": {"mode": "none" if floor is None else ("plane" if floor == FLAT else "surfaces"),
                         "path": None if floor in (None, FLAT) else Path(str(floor)).name,
                         "extrapolated_ids": sorted(ext_ids)},
               "floor_src": {"meaning": TW.FLOOR_SRC_TABLE,
                             "counts": {v: {str(k): int((np.asarray(c["floor_src"]) == k).sum()) for k in (0, 1, 2)}
                                        for v, c in (("fused", raw), ("fused_interp", interp), ("fused_smooth", smooth))}},
               "framerate": fps, "detection": detection, "per_id": per_id,
               "orientation": {"rows": int(len(ori["id"])), "ippe_flip_rejected": int(ori["ippe_flip_rejected"].sum()),
                               "window_s": 0.1, "frame": "world-from-marker, xyzw, marker frame = Calib.obj() (x right, y up, z out of the face)"},
               "gating": {"z_gated_raw": info["z_gated"], "height_gated_raw": n_hgate_raw, "height_gated_smooth": n_hgate_smooth,
                          "dropped_unrefined": info["dropped_unrefined"], "duplicates": info["duplicates"],
                          "too_few_rows": info["too_few_rows"]},
               **sinfo}
    TW.write_atomic(side, (json.dumps(jsonable(sidecar), indent=1, sort_keys=True, allow_nan=False) + "\n").encode())
    TW.write_atomic(stamp, (json.dumps({"inputs_key": key}, indent=1) + "\n").encode())
    return {"skipped": False, "inputs_key": key, "n_raw": int(len(raw["id"])), "n_smooth": int(len(smooth["id"]))}
