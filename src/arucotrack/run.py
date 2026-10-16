"""`arucotrack run config.yaml [key=value ...]`: video -> detection -> refined corners -> pose -> floor -> files.

One process, stage by stage, each stage skipped when its output exists for the same inputs (`--force` redoes them):

1. detect   `detect.run` (multipass ArUco, the `detect` and `marker` groups) -> `<work>/<stem>.nvdec_y.<tag>.npz`
2. refine   `refine_cache.run` (interior-edge corner fit, border constant, quality gate: `refine`, `gate`)
            -> `<work>/<stem>.nvdec_y.<tag>.<gate.mode>.refined.npz`
3. metric   `pipeline.metric_step` (IPPE pose, floor, gates, smoother, interpolation, files: `floor`, `track`)
            -> `<output.dir>/`, plus `<stem>.config.yaml`, the fully resolved configuration

The configuration schema, its defaults and every field's meaning: `arucotrack.config` (`docs/CONFIG.md`). Relative
input paths (video, calibration, surfaces) are taken from `data_root` (default: the config file's directory,
`--data-root` overrides), relative `output.dir` / `output.work` from the current directory. Overrides use dotted
paths: `arucotrack run config.yaml gate.mode=drop camera.frames=120`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import config as C


def _input_root(cfg, config_path: Path | None, data_root: Path | None) -> Path:
    """The directory relative input paths are taken from."""
    if data_root is not None:
        return Path(data_root)
    if cfg.data_root:
        return Path(cfg.data_root) if Path(cfg.data_root).is_absolute() or config_path is None \
            else Path(config_path).parent / cfg.data_root
    return Path(config_path).parent if config_path is not None else Path.cwd()


def _input(base: Path, p) -> Path:
    return Path(p) if Path(p).is_absolute() else base / p


def resolve(cfg, config_path: Path | None, data_root: Path | None = None) -> dict:
    """Absolute input paths of a checked configuration. `floor.model` must already be resolved
    (`config.resolve_auto_floor`, which needs the calibration): an "auto" raises ValueError."""
    if cfg.floor.model == "auto":
        raise ValueError("floor.model is still 'auto': call config.resolve_auto_floor(cfg, calib) first")
    base = _input_root(cfg, config_path, data_root)
    inp = (lambda p: _input(base, p))
    videos = [cfg.video] if isinstance(cfg.video, str) else list(cfg.video)
    floor = {"plane": "plane", "none": None}.get(cfg.floor.model)
    if cfg.floor.model == "surfaces":
        floor = inp(cfg.floor.surfaces)
    return {"data_root": base, "videos": [inp(v) for v in videos], "calibration": inp(cfg.camera.calibration),
            "floor": floor, "work": Path(cfg.output.work), "out": Path(cfg.output.dir)}


def _stage_done(out: Path, stamp: dict) -> bool:
    side = out.with_name(out.name + ".run.json")
    return out.is_file() and side.is_file() and json.loads(side.read_text()).get("stamp") == stamp


def _stage_mark(out: Path, stamp: dict, seconds: float) -> None:
    out.with_name(out.name + ".run.json").write_text(json.dumps({"stamp": stamp, "seconds": round(seconds, 3)},
                                                                 indent=1, sort_keys=True) + "\n")


def _file_id(p: Path) -> dict:
    st = Path(p).stat()
    return {"path": str(Path(p).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _non_default(cfg, group: str, keys: tuple[str, ...]) -> dict:
    """The values of `keys` in `group` that differ from the schema's defaults (so a default run keeps its stamps)."""
    d = C.default()[group]
    return {k: cfg[group][k] for k in keys if cfg[group][k] != d[k]}


def run(cfg, *, config_path: Path | None = None, data_root: Path | None = None, force: bool = False) -> dict:
    """Run detect, refine and the metric step on every video of `cfg`. The caller's OpenCV thread count is restored
    afterwards (`detect.threads` applies only while the run is in progress)."""
    import cv2
    saved = cv2.getNumThreads()
    try:
        return _run(cfg, config_path=config_path, data_root=data_root, force=force)
    finally:
        cv2.setNumThreads(saved)


def _code_stamp() -> dict:
    """The code that makes a stage's output: the package version and its commit (`pipeline.code_state`). A stage
    made by other code is redone."""
    from .pipeline import code_state
    c = code_state()["arucotrack"]
    return {"version": c["version"], "commit": c["git_describe"]}


def _check_inputs(paths: dict, calib, work: Path, tag: str) -> dict:
    """Everything that can be checked before detection, so a bad input fails in seconds, not after hours: one stem
    per video, the same stem not made from another video in `work`, each video's size against the calibration and
    its frame count, and the walking-surfaces file. Returns the probe of each video."""
    from .geometry import surfaces
    from .io import video
    stems: dict[str, Path] = {}
    for vid in paths["videos"]:
        stem = Path(vid).stem
        if stem in stems:
            raise ValueError(f"{stems[stem]} and {vid} have the same file stem {stem!r}, so their outputs would "
                             f"overwrite each other. Rename one, or run them with separate --out and --work.")
        stems[stem] = Path(vid)
    infos = {}
    for stem, vid in stems.items():
        stamp = work / f"{stem}.nvdec_y.{tag}.npz.run.json"
        if stamp.is_file():
            before = (json.loads(stamp.read_text()).get("stamp") or {}).get("video", {}).get("path")
            if before and before != str(Path(vid).resolve()):
                raise ValueError(f"{work} holds the caches of {stem!r} made from {before}, not from {vid}. Use "
                                 f"another --work (and --out) for this video.")
        if not Path(vid).is_file():
            raise FileNotFoundError(f"video {vid} does not exist")
        info = video.probe(vid)
        if (info.width, info.height) != tuple(calib.image_size):
            raise ValueError(f"{vid} is {info.width}x{info.height} but the calibration {calib.source} is for "
                             f"{calib.image_size[0]}x{calib.image_size[1]}: calibrate at the video's resolution")
        if info.n_packets < 2:
            raise ValueError(f"{vid} has {info.n_packets} frame: a trajectory needs at least two frames")
        infos[stem] = info
    if paths["floor"] not in (None, "plane"):
        surfaces.load_surfaces(paths["floor"])
    return infos


def _run(cfg, *, config_path: Path | None, data_root: Path | None, force: bool) -> dict:
    from . import pipeline
    from .calibration.camera import Calib
    from .detection import detect
    from .io import video
    from .refinement import refine_cache
    calib_path = _input(_input_root(cfg, config_path, data_root), cfg.camera.calibration)
    calib = Calib.from_json(calib_path, camera=cfg.camera.name, marker_m=float(cfg.marker.side_m))
    if calib.fisheye:
        raise ValueError(f"{calib_path}: a fisheye calibration. `arucotrack run` needs a pinhole calibration: it "
                         "detects and refines on raw frames and poses with the pinhole model.")
    cfg = C.resolve_auto_floor(cfg, calib)
    C.check(cfg, has_extrinsics=calib.has_extrinsics)
    paths = resolve(cfg, config_path, data_root)
    infos = _check_inputs(paths, calib, paths["work"], C.det_tag(cfg))
    video.set_decoder(cfg.camera.decoder)
    params = pipeline.Params.from_config(cfg)
    max_frames = cfg.camera.frames
    d = {"det_spec": C.det_spec(cfg), "tag": C.det_tag(cfg), "patch": int(cfg.detect.patch)}
    ids = [int(i) for i in cfg.marker.ids] if cfg.marker.ids else None
    marker_extra = {}
    if ids:
        marker_extra["ids"] = ids
    if cfg.marker.dictionary != C.MarkerConfig.dictionary:
        marker_extra["dictionary"] = cfg.marker.dictionary
    refine_extra = {**_non_default(cfg, "refine", ("samples_per_edge", "iterations", "cauchy_c", "crop_px",
                                                   "blur_window", "exposure_fraction", "sigma_optics")),
                    **_non_default(cfg, "gate", ("max_edge_rms_px", "min_shift_px", "max_shift_frac",
                                                 "max_downweighted_frac", "agree_px", "window_frames"))}
    work, out = paths["work"], paths["out"]
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    camera = cfg.camera.name or calib.camera
    line = C.summary(cfg)
    code = _code_stamp()
    report = {}
    for vid in paths["videos"]:
        stem = Path(vid).stem
        (out / f"{stem}.config.yaml").write_text(
            f"# arucotrack {code['version']}: the resolved configuration of {stem}\n"
            "# relative inputs are taken from --data-root (or this file's folder)\n" + C.to_yaml(cfg))
        timing = {}
        # 1. detect
        det_cache = work / f"{stem}.nvdec_y.{d['tag']}.npz"
        stamp = {"video": _file_id(vid), "detect": d, "decoder": cfg.camera.decoder, "max_frames": max_frames,
                 "code": code, **marker_extra}
        t0 = time.perf_counter()
        if force or not _stage_done(det_cache, stamp):
            n_threads = detect.set_threads(cfg.detect.threads)
            logger.info("{}: detect ({}), {} OpenCV threads, decoder {}", stem, d["det_spec"], n_threads,
                        video.decoder_label("nvdec_y", infos[stem].codec))
            detect.run(str(Path(vid).resolve()), source="nvdec_y", patch=int(d["patch"]), tag=d["tag"],
                       cache_root=work, det_spec=d["det_spec"], stem=stem, max_frames=max_frames, ids=ids,
                       dictionary=cfg.marker.dictionary)
            _stage_mark(det_cache, stamp, time.perf_counter() - t0)
        timing["detect_s"] = round(time.perf_counter() - t0, 3)
        with np.load(det_cache, allow_pickle=False) as cs:
            if len(cs["det_id"]) == 0:
                logger.info("no markers detected in {}: the output files hold their headers only", vid)
        # 2. refine
        mode = cfg.gate.mode
        refined = work / f"{stem}.nvdec_y.{d['tag']}.{mode}.refined.npz"
        stamp = {"detect_cache": _file_id(det_cache), "calibration": calib.sha256,
                 "border_cells": float(cfg.marker.border_cells), "gate_mode": mode, "decoder": cfg.camera.decoder,
                 "max_frames": max_frames, "code": code, **refine_extra,
                 **({"dictionary": marker_extra["dictionary"]} if "dictionary" in marker_extra else {})}
        t0 = time.perf_counter()
        if force or not _stage_done(refined, stamp):
            logger.info("{}: refine (border_cells {}, gate {})", stem, cfg.marker.border_cells, mode)
            part = refined.with_name(refined.name.replace(".refined.npz", ".part.refined.npz"))
            part.unlink(missing_ok=True)
            r, g = cfg.refine, cfg.gate
            refine_cache.run(det_cache, calib=calib, video=Path(vid), camera=camera or None, out=part,
                             border_cells=float(cfg.marker.border_cells), source="nvdec_y", gate_mode=mode,
                             max_frames=max_frames, crop=int(r.crop_px),
                             max_edge_rms_px=float(g.max_edge_rms_px), max_shift_frac=float(g.max_shift_frac),
                             min_shift_px=float(g.min_shift_px), max_downweighted_frac=float(g.max_downweighted_frac),
                             gate_agree_px=float(g.agree_px), gate_window_frames=float(g.window_frames),
                             blur_window=bool(r.blur_window), exposure_fraction=r.exposure_fraction,
                             sigma_optics=r.sigma_optics, samples_per_edge=int(r.samples_per_edge),
                             iterations=int(r.iterations), cauchy_c=float(r.cauchy_c),
                             dictionary=cfg.marker.dictionary)
            part.replace(refined)
            pj = part.with_suffix(".json")
            if pj.is_file():
                pj.replace(refined.with_suffix(".json"))
            _stage_mark(refined, stamp, time.perf_counter() - t0)
        timing["refine_s"] = round(time.perf_counter() - t0, 3)
        # 3. metric
        t0 = time.perf_counter()
        res = pipeline.metric_step(stem, refined, det_cache, out, work, calib=calib, floor=paths["floor"],
                                   params=params, camera=camera, force=force, config_line=line)
        timing["metric_s"] = round(time.perf_counter() - t0, 3)
        report[stem] = {**res, "timing": timing}
        logger.info("{}: {} -> {} ({})", stem, res, out, timing)
    return report


# Short flags: each is the dotted override named here.
FLAG_OVERRIDES = {"work": "output.work", "out": "output.dir"}


def parse(argv=None) -> argparse.Namespace:
    """The command line. Dotted overrides may come before or after the options (intermixed)."""
    ap = argparse.ArgumentParser(prog="arucotrack run", description=__doc__.splitlines()[0],
                                 epilog="Configuration fields: docs/CONFIG.md")
    ap.add_argument("config", type=Path, help="the YAML configuration")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides, e.g. gate.mode=drop camera.frames=120")
    ap.add_argument("--data-root", type=Path, default=None, metavar="DIR",
                    help="base of the config's relative input paths")
    ap.add_argument("--out", default=None, metavar="DIR", help="the same as output.dir=...")
    ap.add_argument("--work", default=None, metavar="DIR", help="the same as output.work=...")
    ap.add_argument("--force", action="store_true", help="redo every stage")
    ap.add_argument("--debug", action="store_true", help="show the full traceback of an unexpected error")
    return ap.parse_intermixed_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    overrides = list(a.overrides) + [f"{key}={getattr(a, k)}" for k, key in FLAG_OVERRIDES.items()
                                     if getattr(a, k) is not None]
    try:
        cfg = C.load(a.config, overrides)
        rep = run(cfg, config_path=a.config, data_root=a.data_root, force=a.force)
    except (ValueError, FileNotFoundError) as e:
        print(f"arucotrack run: {e}", file=sys.stderr)
        return 2
    except Exception as e:                              # decoder, library or file-system errors: one clean line
        if a.debug:
            raise
        print(f"arucotrack run: {type(e).__name__}: {e} (--debug shows the traceback)", file=sys.stderr)
        return 1
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
