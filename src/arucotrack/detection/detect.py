"""The detection cache: one pass over a video, raw observations only.

Per detection: packet sequence number and PTS, marker id, the four raw corners, a local sharpness, and optionally a
uint8 patch around the marker (`detect.patch`). The rejected candidate quads (what the detector found but could not
decode) are counted per run. Patches serve the inspection of a cache. No stage of this package reads them, and they
grow with the length of the video. Derived quantities (side, radius, velocity, pose) are computed from the cache
afterwards, so a changed calibration or threshold never requires another pass over the video.

The detector is the one `det_spec` names (`detectors.parse_spec`). The full parameter set of its base
`DetectorParameters` is recorded in the provenance.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .. import provenance
from ..io.video import Source, decoder_label, frames, pixel_pipeline_label, probe


def make_params() -> cv2.aruco.DetectorParameters:
    """OpenCV's default detector parameters with `CORNER_REFINE_SUBPIX`: the base every detector pass starts from,
    recorded in full in the cache's provenance."""
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # the baseline: OpenCV's sub-pixel corners
    return par


def _sharpness(gray: np.ndarray, quad: np.ndarray) -> float:
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 1
    h, w = gray.shape
    roi = gray[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
    if roi.size < 16:
        return float("nan")
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _patch(gray: np.ndarray, centre: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    """Zero-padded size x size crop centred on the marker, and its origin."""
    h, w = gray.shape
    cx, cy = int(round(float(centre[0]))), int(round(float(centre[1])))
    x0, y0 = cx - size // 2, cy - size // 2
    out = np.zeros((size, size), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    return out, x0, y0


def peak_rss_mib() -> float:
    """Peak resident memory of this process, MiB (`ru_maxrss` is in KiB on Linux and in bytes on macOS)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024


def resolve_threads(n: int) -> int:
    """`detect.threads` as a thread count: 0 = all usable cores (`os.sched_getaffinity`, else `os.cpu_count`),
    a positive value as given, a negative one an error."""
    n = int(n)
    if n < 0:
        raise ValueError(f"detect.threads must be 0 (all usable cores) or a positive number, not {n}")
    if n:
        return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def set_threads(n: int) -> int:
    """Apply `detect.threads` to OpenCV (the one place that does) and return the thread count in use."""
    k = resolve_threads(n)
    cv2.setNumThreads(k)
    return k


def run(video: str | Path, *, cache_root: Path, det_spec: str, tag: str, stem: str, source: Source = "nvdec_y",
        patch: int = 0, max_frames: int | None = None, ids: list[int] | None = None,
        dictionary: str | int | None = None) -> dict[str, Any]:
    """One pass over `video` (a path, absolute or relative to the working directory) -> `<cache_root>/<stem>.<source>.
    <tag>.npz` and its `.json` sidecar. `det_spec` (e.g. 'multipass:subpix ac=3,6,9 scales=1,2 roster') is the
    `detectors.parse_spec` detector. `ids` (the marker ids in use, given by the caller) is the roster. The
    off-roster counts are reported only with it, and a det_spec's `roster` restricts the dictionary to it.
    `dictionary` (a `cv2.aruco` name or constant, default MIP_36h12) is the detector's dictionary. The rejected
    candidate quads are counted in `summary.rejected_candidates`, not stored."""
    started = _dt.datetime.now(_dt.timezone.utc)
    path = Path(video)
    if not path.exists():
        raise FileNotFoundError(path)
    info = probe(path)
    if info.n_packets == 0:
        raise ValueError(f"{path.name}: the video has no packets (frame_pts_all would be empty); refusing to write "
                         f"an empty cache")
    n_range = min(max_frames, info.n_packets) if max_frames else info.n_packets
    par = make_params()
    from . import detectors as _bd
    dict_id = _bd.dictionary_id(dictionary)
    ros, ros_raw = ({int(i) for i in ids}, [str(int(i)) for i in ids]) if ids else (set(), [])
    spec_det = _bd.parse_spec(det_spec, sorted(ros) if ros else None, dictionary=dict_id)
    timing = info.timing()
    logger.info("{}: {}x{} {} {} @ {} fps nominal, {} packets over {} s ({} pk/s, {} nominal slots missing); source={} tag={}",
                path.name, info.width, info.height, info.codec, info.pix_fmt, info.fps,
                info.n_packets, timing["span_s"], timing["packets_per_span_fps"], timing["nominal_slots_missing"], source, tag)

    d_seq, d_pts, d_id, d_corners, d_sharp, d_patch, d_origin = [], [], [], [], [], [], []
    n_rejected = 0
    n_frames = 0
    decoded = frames(path, source, max_frames=(n_range if max_frames else None), info=info)
    for fr in tqdm(decoded, total=n_range, unit="fr", file=sys.stderr, mininterval=1.0):
        n_frames += 1
        g = fr.gray
        dets, rej = spec_det.detect_with_rejected(g)
        corners = np.asarray([d.corners for d in dets], np.float32).reshape(-1, 4, 2)
        for q, mid in zip(corners, (d.id for d in dets)):
            d_seq.append(fr.seq); d_pts.append(fr.pts); d_id.append(int(mid))
            d_corners.append(q); d_sharp.append(_sharpness(g, q))
            if patch:
                p, x0, y0 = _patch(g, q.mean(0), patch)
                d_patch.append(p); d_origin.append((x0, y0))
        n_rejected += len(rej)

    ids_arr = np.asarray(d_id, np.int32)
    uniq, counts = np.unique(ids_arr, return_counts=True)
    # With no roster the off-roster count is unknown, not zero.
    off = sorted(int(u) for u in uniq if int(u) not in ros) if ros else None
    maxrss_mib = peak_rss_mib()
    runtime = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
    summary: dict[str, Any] = {
        "video": path.name, "source": source, "tag": tag,
        "frames_decoded": n_frames, "packets": info.n_packets, "timing": timing,
        "detections": int(len(ids_arr)), "frames_with_detection": int(len(np.unique(d_seq))),
        "distinct_ids": int(len(uniq)), "ids": {int(u): int(c) for u, c in zip(uniq, counts)},
        "roster_known": bool(ros),
        "off_roster_ids": off,
        "off_roster_detections": (int(sum(int(c) for u, c in zip(uniq, counts) if int(u) not in ros)) if ros else None),
        "rejected_candidates": int(n_rejected), "rejected_per_frame": round(n_rejected / max(n_frames, 1), 2),
        "patch_px": patch if patch else 0, "det_spec": det_spec,
        "multipass_conflicts": int(getattr(spec_det, "conflicts", 0)),
        "runtime_s": round(runtime, 1), "fps": round(n_frames / max(runtime, 1e-9), 1), "peak_rss_mib": round(maxrss_mib),
    }
    # an absolute path is recorded by its file name only: a machine path does not belong in a shared cache
    prov_rel = str(Path(video)) if not Path(video).is_absolute() else path.name
    prov = provenance.block(relpath=prov_rel,
                            decoder=decoder_label(source, info.codec), pixel_pipeline=pixel_pipeline_label(source, info.pix_fmt),
                            params={"dictionary": _bd.dictionary_name(dict_id), "detector": provenance.detector_params_dict(par),
                                    "patch_px": patch, "max_frames": max_frames,
                                    "det_spec": det_spec, "det_name": spec_det.name,
                                    "roster_marker_ids_raw": ros_raw},
                            started=started, extra={"summary": summary})

    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    npz = cache_root / f"{stem}.{source}.{tag}.npz"
    arrays = dict(
        det_seq=np.asarray(d_seq, np.int64), det_pts=np.asarray(d_pts, np.float64),
        det_id=ids_arr, det_corners=np.asarray(d_corners, np.float32).reshape(-1, 4, 2),
        det_sharpness=np.asarray(d_sharp, np.float64),
        frame_pts_all=info.pts,  # time of every packet, so seq -> pts is always recoverable
        provenance=np.array(json.dumps(prov)),
    )
    if patch:
        arrays["det_patch"] = np.asarray(d_patch, np.uint8).reshape(-1, patch, patch)
        arrays["det_patch_origin"] = np.asarray(d_origin, np.int32).reshape(-1, 2)
    _write_cache(npz, arrays, prov)
    summary["cache"] = str(npz)
    summary["cache_mib"] = round(npz.stat().st_size / 2**20, 1)
    logger.info("wrote {} ({} MiB): {} detections, {} ids, {} off-roster, {:.1f} fps, peak RSS {} MiB",
                npz.name, summary["cache_mib"], summary["detections"], summary["distinct_ids"],
                summary["off_roster_detections"], summary["fps"], summary["peak_rss_mib"])
    return summary


def _write_cache(npz: Path, arrays: dict[str, np.ndarray], prov: dict) -> None:
    """`<x>.npz` and its `<x>.json` sidecar, each written under a temporary name in the same directory and
    renamed: a killed pass never leaves a truncated cache under the real name. The sidecar is renamed last,
    so a `.json` next to an `.npz` means both are complete.

    The sidecar is the embedded provenance plus `npz_sha256`, the sha256 of the npz it was written with. A kill
    between the two renames leaves a new npz beside an older sidecar, which `refine_cache` notices by the file
    times and then hashes the npz itself."""
    side = npz.with_suffix(".json")
    tmp_npz, tmp_json = npz.with_name(npz.name + ".tmp"), side.with_name(side.name + ".tmp")
    with open(tmp_npz, "wb") as fh:                 # a file handle: savez would append .npz to a bare name
        np.savez_compressed(fh, **arrays)
    tmp_json.write_text(json.dumps({**prov, "npz_sha256": provenance.sha256_file(tmp_npz)}, indent=1))
    os.replace(tmp_npz, npz)
    os.replace(tmp_json, side)


class Arrays(dict):
    """The arrays of an npz cache read into memory, with `files` like an open `NpzFile`."""

    @property
    def files(self) -> list[str]:
        return list(self)


def load_arrays(path: Path, skip: tuple[str, ...] = ("det_patch",)) -> Arrays:
    """The arrays of a cache, without pickled objects and without the arrays named by a prefix in `skip` (patches,
    which no stage reads). The file is closed on return."""
    with np.load(Path(path), allow_pickle=False) as z:
        return Arrays({k: z[k] for k in z.files if not k.startswith(skip)})
