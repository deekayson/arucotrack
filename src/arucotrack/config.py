"""One configuration for a run: the schema, its defaults, loading, overrides and checks.

The schema is a set of dataclasses read by OmegaConf as a structured config, so a YAML file is checked against it
(unknown keys and wrong types are errors) and `key=value` overrides on the command line use dotted paths
(`gate.mode=drop`, `camera.frames=120`, `marker.ids=[17,18]`). The defaults are the values of the paper (Appendix A,
"Parameters of the tracker") except where a field docstring says otherwise. Every field carries a docstring, from
which `docs/CONFIG.md` is generated (`markdown`).

Groups:

    marker  the printed marker: dictionary, side, border constant, optional id roster
    camera  the calibration file, the decoder and an optional frame limit
    floor   the floor model: `plane` (z = 0), `surfaces` (a walking-surfaces JSON) or `none`
    detect  the multipass ArUco detector
    refine  the interior-edge corner refinement
    gate    the refinement quality gate and what happens to a refused refinement
    track   pose gates, segments, smoother, interpolation
    output  where caches and files go, and which files are written

The fully resolved configuration is written as `<output.dir>/<stem>.config.yaml` next to the files it produced, and
its one-line summary (`summary`) is recorded in the sidecar.

Fields marked *fixed* are part of the method as published: they are listed so the configuration documents every
number, and the loader refuses a value other than the default (the code does not take them as parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import MISSING, OmegaConf

# ---------------------------------------------------------------------------- schema


@dataclass
class MarkerConfig:
    dictionary: str = "DICT_ARUCO_MIP_36h12"
    """OpenCV predefined ArUco dictionary, by its `cv2.aruco` name. The refinement models a 6x6 bit grid inside a
    one-cell border, so only 6x6 dictionaries are accepted."""
    side_m: float = MISSING
    """Side of the printed black square, metres (required: measure your print). Sets the scale of every position."""
    border_cells: float = 0.869
    """Printed border width in data-cell widths, nominally 1. The default is for an unmeasured print: set the value
    for your print and camera against a marker at a known position (docs/CALIBRATION.md, "Border constant")."""
    ids: Optional[List[int]] = None
    """Optional roster: the marker ids in use. The dictionary is restricted to them, so other codes cannot decode and
    more bit errors can be corrected without ambiguity."""


@dataclass
class CameraConfig:
    calibration: str = MISSING
    """Calibration JSON (`docs/CALIBRATION_FILE.md`): intrinsics and an optional camera pose. Without a camera pose
    the positions are in the camera frame, and `floor.model: auto` switches the floor and both height gates off."""
    name: str = ""
    """Camera id recorded in the sidecar (default: the calibration file's `camera_id`)."""
    decoder: str = "auto"
    """Video decoder: `auto` (NVDEC when a GPU and ffmpeg's CUVID decoder are present, else software), `nvdec`,
    `sw`."""
    frames: Optional[int] = None
    """Process only the first N frames (a quick check), null = the whole video."""


@dataclass
class FloorConfig:
    model: str = "auto"
    """`auto`: `plane` with a camera pose, else `none` with both height gates off. `plane` (the plane z = 0),
    `surfaces` (the walking surfaces of `floor.surfaces`) and `none` (no floor columns) can be set, and `plane` and
    `surfaces` need a camera pose."""
    surfaces: Optional[str] = None
    """Walking-surfaces JSON, for `floor.model: surfaces` only (`docs/FORMATS.md`, "Walking surfaces")."""


@dataclass
class DetectConfig:
    constants: List[int] = field(default_factory=lambda: [3, 6, 9])
    """Adaptive-threshold constants of the multipass detector. Every pass runs and the passes are merged."""
    scales: List[float] = field(default_factory=lambda: [1.0, 2.0])
    """Image scales of the multipass detector (2 = the frame upscaled 2x, bicubic)."""
    win_size_max: int = 73
    """Adaptive-threshold window maximum, px (`adaptiveThreshWinSizeMax`)."""
    win_size_step: int = 4
    """Adaptive-threshold window step, px (`adaptiveThreshWinSizeStep`)."""
    pixels_per_cell: int = 8
    """Perspective removal pixels per cell (`perspectiveRemovePixelPerCell`)."""
    corner_refinement: str = "subpix"
    """OpenCV corner refinement of each pass: `subpix`, `none`, `contour` or `apriltag`."""
    iou: float = 0.5
    """Intersection over union above which detections of two passes are one marker."""
    periphery_rule: str = "any_corner"
    """*fixed*. Not used by `arucotrack run` (pinhole cameras only)."""
    theta_max_deg: float = 65.0
    """*fixed*. Not used by `arucotrack run` (pinhole cameras only)."""
    threads: int = 0
    """OpenCV threads during detection, 0 = all usable cores, the cores this process may run on (on a shared machine
    set a number). Detections are byte-identical for any value."""
    patch: int = 0
    """Marker image patches stored in the detection cache, px (0 = none)."""


@dataclass
class RefineConfig:
    samples_per_edge: int = 3
    """Samples per interior edge."""
    inset_cells: float = 0.22
    """*fixed*. Inset of the samples from the edge ends, cells."""
    half_window: str = "0.42 cell, clipped to 1-4 px"
    """*fixed*. Half-width of each edge profile: 0.42 of a cell, clipped to 1-4 px."""
    profile_step_px: float = 0.25
    """*fixed*. Edge profile sample spacing, px."""
    min_contrast: float = 12.0
    """*fixed*. Minimum edge contrast, grey levels."""
    cauchy_c: float = 1.5
    """Cauchy constant of the robust line fit, in units of the robust scale."""
    gauss_newton_steps: int = 6
    """*fixed*. Gauss-Newton steps per outer iteration."""
    iterations: int = 3
    """Outer iterations (re-sample the edges, re-fit the homography)."""
    min_samples: int = 12
    """*fixed*. Minimum usable edge samples. With fewer, the refinement fails."""
    max_move_cells: float = 1.0
    """*fixed*. A refinement that moves the corners by more than this (cells) has diverged."""
    crop_px: int = 224
    """Square crop around each marker in which the refinement works, px."""
    blur_window: bool = False
    """Blur-adaptive window: widen each edge profile by the motion blur `exposure_fraction x |v . n|` of a moving
    marker. Off by default (the values used in the paper: `docs/METHOD.md`)."""
    exposure_fraction: Optional[float] = None
    """Effective exposure fraction for the blur window (null = fitted from the video's fast markers)."""
    sigma_optics: Optional[float] = None
    """At-rest optics blur, px (required with `blur_window`)."""
    velocity_window_frames: float = 2.5
    """*fixed*. Neighbouring detections within this many frame intervals give a marker's image velocity."""


@dataclass
class GateConfig:
    mode: str = "resolve"
    """What happens to a refused refinement. `resolve`: the detector corners or the refused refinement are used when
    they agree with accepted neighbours on both sides, else the row is left out of the trajectory files. `drop`:
    the row is removed."""
    max_edge_rms_px: float = 0.8
    """Largest edge root-mean-square residual of an accepted refinement, px."""
    min_shift_px: float = 1.5
    """Lower bound of the corner shift limit: an accepted refinement moves the corners by at most
    max(min_shift_px, max_shift_frac x diagonal), px."""
    max_shift_frac: float = 0.07
    """Corner shift limit as a fraction of the detector quad's diagonal."""
    max_downweighted_frac: float = 0.3
    """Largest share of unusable or far edge samples."""
    agree_px: float = 3.0
    """`resolve`: largest corner distance from the position predicted by the accepted neighbours, px."""
    window_frames: float = 5.0
    """`resolve`: the accepted neighbours must lie within this many frames on both sides."""


@dataclass
class TrackConfig:
    z_range_m: Optional[List[float]] = field(default_factory=lambda: [-3.0, 4.5])
    """World height gate on the marker centre [min, max], metres, applied before the per-frame de-duplication
    (null = off). Needs a camera pose."""
    height_band_m: Optional[List[float]] = field(default_factory=lambda: [0.3, 2.8])
    """Marker height above the floor surface under it [min, max], metres, on rows that have a surface (null = off).
    Needs a camera pose and a floor."""
    max_gap_frames: int = 30
    """Largest gap inside a segment, frames. A longer gap splits the track."""
    min_segment_rows: int = 5
    """Shortest segment kept in the smoothed files, measured rows."""
    sigma_acc_m: float = 0.00045
    """Acceleration prior of the smoother, metres per frame squared (each axis). The default is the RMS head
    acceleration of walking participants below 6 Hz (the paper, Section 3.6)."""
    sigma_s: str = "1.4826 MAD of second differences / sqrt(6), pooled; 2 mm below 20 triples; floor 0.1 mm"
    """Measurement noise of the smoother, metres (each axis), for example `0.0023`, one value for every measured row
    of the clip. The default text estimates it per clip as it says."""
    interp_cap_frames: int = 30
    """Largest gap filled by linear interpolation, frames."""


@dataclass
class OutputConfig:
    dir: str = "out"
    """Trajectory files, relative to the current directory (`--out` overrides it)."""
    work: str = "work"
    """Caches of detection, refined corners and poses, relative to the current directory (`--work` overrides it)."""
    dataset: str = "dataset"
    """Dataset label, kept in the resolved configuration (`<stem>.config.yaml`)."""
    variants: List[str] = field(default_factory=lambda: ["fused", "fused_interp", "fused_smooth", "petrack"])
    """*fixed*. Files written per video: measured rows, plus linear fill, smoothed, PeTrack six-column."""


@dataclass
class Config:
    video: List[str] = MISSING
    """Video file(s), relative to `data_root`."""
    data_root: Optional[str] = None
    """Base of relative input paths (default: the config file's directory, `--data-root` overrides)."""
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    floor: FloorConfig = field(default_factory=FloorConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


GATE_MODES = ("resolve", "drop")
MIN_SIDE_M, MAX_SIDE_M = 0.01, 2.0       # plausible printed marker sides, metres: catches centimetres typed for metres
DECODERS = ("auto", "nvdec", "sw")
FLOOR_MODELS = ("auto", "plane", "surfaces", "none")
CORNER_REFINEMENTS = ("none", "subpix", "contour", "apriltag")

# ---------------------------------------------------------------------------- loading


def default() -> Any:
    """The schema with its defaults, an OmegaConf DictConfig (`video`, `camera.calibration` and `marker.side_m` are
    missing)."""
    return OmegaConf.structured(Config)


def load(path: str | Path | None = None, overrides: list[str] | tuple[str, ...] = ()) -> Any:
    """Schema <- YAML file <- dotted `key=value` overrides, then `check`. Raises ValueError with the offending key."""
    from omegaconf.errors import OmegaConfBaseException
    try:
        cfg = default()
        if path is not None:
            user = OmegaConf.load(Path(path))
            if isinstance(user.get("video"), str):          # one video may be given as a plain string
                user.video = [user.video]
            cfg = OmegaConf.merge(cfg, user)
        if overrides:
            over = OmegaConf.from_dotlist(list(overrides))
            if isinstance(over.get("video"), str):          # `video=a.mp4` on the command line
                over.video = [over.video]
            cfg = OmegaConf.merge(cfg, over)
    except OmegaConfBaseException as e:
        key = getattr(e, "full_key", None)
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        raise ValueError(f"{path or 'config'}: {key + ': ' if key else ''}{msg}") from None
    check(cfg)
    return cfg


def _fixed_fields() -> list[tuple[str, Any]]:
    """(dotted key, default) of every field documented as *fixed*."""
    out = []
    for group, typ in _groups():
        defaults = OmegaConf.structured(typ)
        for f in fields(typ):
            if (_doc(typ, f.name) or "").startswith("*fixed*"):
                out.append((f"{group}.{f.name}", OmegaConf.select(defaults, f.name)))
    return out


def _groups() -> list[tuple[str, type]]:
    return [(g.name, g.default_factory) for g in fields(Config)
            if callable(getattr(g, "default_factory", None)) and is_dataclass(g.default_factory)]


def check(cfg: Any, *, has_extrinsics: bool | None = None) -> None:
    """Values the schema's types cannot express. `has_extrinsics` (from the calibration file, when known) checks
    that the floor model and the height gates have a world frame to work in."""
    import cv2
    if OmegaConf.is_missing(cfg, "video") or not cfg.video:
        raise ValueError("`video` is required (one path or a list)")
    if OmegaConf.is_missing(cfg.camera, "calibration") or not cfg.camera.calibration:
        raise ValueError("`camera.calibration` is required (a calibration JSON, docs/CALIBRATION_FILE.md)")
    if OmegaConf.is_missing(cfg.marker, "side_m"):
        raise ValueError("`marker.side_m` is required (the side of your printed black square, metres)")
    name = cfg.marker.dictionary
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        known = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))
        raise ValueError(f"marker.dictionary {name!r} is not a cv2.aruco predefined dictionary ({', '.join(known)})")
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    if dic.markerSize != 6:
        raise ValueError(f"marker.dictionary {name}: the refinement models a 6x6 bit grid. Use a 6x6 dictionary.")
    if not MIN_SIDE_M <= cfg.marker.side_m <= MAX_SIDE_M:
        raise ValueError(f"marker.side_m {cfg.marker.side_m}: the side of the printed black square in metres, "
                         f"between {MIN_SIDE_M} and {MAX_SIDE_M} (0.18 is 18 cm)")
    if not 0.5 < cfg.marker.border_cells < 1.5:
        raise ValueError(f"marker.border_cells {cfg.marker.border_cells}: must lie in (0.5, 1.5) data cells")
    if cfg.marker.ids:
        ids = [int(i) for i in cfg.marker.ids]
        n = len(dic.bytesList)
        if len(set(ids)) != len(ids):
            raise ValueError(f"marker.ids {ids}: every id must be unique")
        if any(not 0 <= i < n for i in ids):
            raise ValueError(f"marker.ids {ids}: {name} has the ids 0..{n - 1}")
    if cfg.camera.decoder not in DECODERS:
        raise ValueError(f"camera.decoder {cfg.camera.decoder!r}: one of {DECODERS}")
    if cfg.camera.frames is not None and cfg.camera.frames < 1:
        raise ValueError("camera.frames must be >= 1 or null")
    if cfg.floor.model not in FLOOR_MODELS:
        raise ValueError(f"floor.model {cfg.floor.model!r}: one of {FLOOR_MODELS}")
    if (cfg.floor.model == "surfaces") != bool(cfg.floor.surfaces):
        raise ValueError("floor.surfaces is given exactly when floor.model is `surfaces`")
    if cfg.gate.mode not in GATE_MODES:
        raise ValueError(f"gate.mode {cfg.gate.mode!r}: one of {GATE_MODES}")
    if cfg.detect.corner_refinement not in CORNER_REFINEMENTS:
        raise ValueError(f"detect.corner_refinement {cfg.detect.corner_refinement!r}: one of {CORNER_REFINEMENTS}")
    if cfg.detect.threads < 0:
        raise ValueError(f"detect.threads {cfg.detect.threads}: 0 (all usable cores) or a positive count")
    if not cfg.detect.constants or not cfg.detect.scales:
        raise ValueError("detect.constants and detect.scales need at least one value each")
    if cfg.refine.blur_window and cfg.refine.sigma_optics is None:
        raise ValueError("refine.blur_window needs refine.sigma_optics (the camera's at-rest optics blur, px)")
    if not cfg.track.sigma_acc_m > 0:
        raise ValueError(f"track.sigma_acc_m {cfg.track.sigma_acc_m}: the acceleration prior must be > 0 (metres per "
                         f"frame squared)")
    for key in ("max_gap_frames", "interp_cap_frames", "min_segment_rows"):
        if cfg.track[key] < 0:
            raise ValueError(f"track.{key} {cfg.track[key]}: must be >= 0 (frames or rows)")
    for key in ("z_range_m", "height_band_m"):
        v = cfg.track[key]
        if v is not None and (len(v) != 2 or not v[0] < v[1]):
            raise ValueError(f"track.{key} must be [min, max] with min < max, or null")
    sigma_s_given(cfg.track.sigma_s)
    for key, value in _fixed_fields():
        if OmegaConf.select(cfg, key) != value:
            raise ValueError(f"{key} is fixed in this release (the code does not take it as a parameter); "
                             f"it must stay {value!r}")
    if has_extrinsics is False:
        bad = [k for k, on in (("floor.model=none", cfg.floor.model != "none"),
                               ("track.z_range_m=null", cfg.track.z_range_m is not None),
                               ("track.height_band_m=null", cfg.track.height_band_m is not None)) if on]
        if bad:
            raise ValueError(f"{cfg.camera.calibration}: the calibration has no extrinsics, so positions are in the "
                             f"camera frame and there is no floor or world height; set {' '.join(bad)}")


# ---------------------------------------------------------------------------- derived values


SIGMA_S_MAX_M = 0.05          # a fixed measurement noise above 5 cm is not a marker measurement


def sigma_s_given(value) -> float | None:
    """`track.sigma_s` -> metres, or None for the default text (the per-clip estimate). ValueError otherwise."""
    if value is None or str(value) == TrackConfig.sigma_s:
        return None
    try:
        v = float(str(value).strip())
    except ValueError:
        raise ValueError(f"track.sigma_s {value!r}: a measurement noise in metres (e.g. 0.0023), or the default "
                         f"(estimated per clip)") from None
    if not 0.0 < v <= SIGMA_S_MAX_M:
        raise ValueError(f"track.sigma_s {value!r}: in (0, {SIGMA_S_MAX_M}] m")
    return v


def resolve_auto_floor(cfg: Any, calib: Any) -> Any:
    """A copy of `cfg` with `floor.model: auto` replaced by what the calibration allows: `plane` with extrinsics,
    `none` without, together with `track.z_range_m` and `track.height_band_m` set to null (there is no world frame).
    Any explicit floor model, and every gate, is returned unchanged (`check` then refuses a mismatch)."""
    out = OmegaConf.create(OmegaConf.to_container(cfg))
    if out.floor.model == "auto":
        if calib.has_extrinsics:
            out.floor.model = "plane"
        else:
            out.floor.model = "none"
            out.track.z_range_m = None
            out.track.height_band_m = None
    return out


def det_spec(cfg: Any) -> str:
    """The detector spec `detectors.parse_spec` reads, from `detect` and `marker.ids`."""
    d = cfg.detect
    spec = (f"multipass:{d.corner_refinement} ac={','.join(str(int(c)) for c in d.constants)} "
            f"scales={','.join(f'{float(s):g}' for s in d.scales)}")
    extra = []
    if d.iou != 0.5:
        extra.append(f"iou={d.iou}")
    if (d.win_size_max, d.win_size_step, d.pixels_per_cell) != (73, 4, 8):
        extra += [f"adaptiveThreshWinSizeMax={d.win_size_max}", f"adaptiveThreshWinSizeStep={d.win_size_step}",
                  f"perspectiveRemovePixelPerCell={d.pixels_per_cell}"]
    if cfg.marker.ids:
        extra.append("roster")
    return " ".join([spec, *extra])


def det_tag(cfg: Any) -> str:
    """Cache tag of a detection run: `multipass_ac369_s12` for the defaults."""
    d = cfg.detect
    tag = (f"multipass_ac{''.join(str(int(c)) for c in d.constants)}"
           f"_s{''.join(f'{float(s):g}'.replace('.', 'p') for s in d.scales)}")
    if cfg.marker.ids:
        tag += "_roster"
    if cfg.marker.dictionary != MarkerConfig.dictionary:
        tag += "_" + cfg.marker.dictionary[5:].lower()
    return tag


def to_yaml(cfg: Any) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def summary(cfg: Any) -> str:
    """One line, recorded in the sidecar: the values that change numbers."""
    m, g, t, r = cfg.marker, cfg.gate, cfg.track, cfg.refine
    return (f"marker {m.dictionary} side {m.side_m} m border {m.border_cells} | "
            f"detect ac {list(cfg.detect.constants)} scales {list(cfg.detect.scales)} | refine samples "
            f"{r.samples_per_edge} cauchy {r.cauchy_c} iterations {r.iterations} blur_window {r.blur_window} | gate "
            f"{g.mode} edge_rms {g.max_edge_rms_px} agree {g.agree_px} px | floor {cfg.floor.model} | track z "
            f"{t.z_range_m} height {t.height_band_m} gap {t.max_gap_frames} min_segment {t.min_segment_rows} "
            f"sigma_acc {t.sigma_acc_m} interp {t.interp_cap_frames}"
            + ("" if t.sigma_s == TrackConfig.sigma_s else f" sigma_s {t.sigma_s}"))


# ---------------------------------------------------------------------------- documentation


def _doc(cls, name: str) -> str | None:
    """The docstring under a dataclass field (the string literal after the annotation)."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    body = tree.body[0].body
    for i, node in enumerate(body):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            if i + 1 < len(body) and isinstance(body[i + 1], ast.Expr) and isinstance(body[i + 1].value, ast.Constant):
                return " ".join(str(body[i + 1].value.value).split())
    return None


def markdown() -> str:
    """docs/CONFIG.md: every field, its default and its meaning, from the dataclasses."""
    lines = ["# Configuration", "",
             "Generated from `src/arucotrack/config.py`. Do not edit by hand.",
             "", " ".join((__doc__ or "").split("\n\n")[0].split()), "",
             "Run with `arucotrack run config.yaml [key=value ...]`. Overrides use dotted paths, for example "
             "`gate.mode=drop camera.frames=120`. The resolved configuration is written next to the output files "
             "(`<stem>.config.yaml`).", ""]
    top = [f for f in fields(Config) if f.name in ("video", "data_root")]
    lines += ["## Top level", "", "| key | default | meaning |", "|---|---|---|"]
    for f in top:
        d = "required" if f.default == MISSING else ("null" if f.default is None else f"`{f.default}`")
        lines.append(f"| `{f.name}` | {d} | {_doc(Config, f.name)} |")
    for group, typ in _groups():
        lines += ["", f"## `{group}`", "", "| key | default | meaning |", "|---|---|---|"]
        for f in fields(typ):
            if callable(f.default_factory):
                shown = f"`{f.default_factory()}`"
            elif f.default is MISSING or f.default == MISSING:
                shown = "required"
            elif f.default is None:
                shown = "null"
            elif f.default == "":
                shown = "empty"
            else:
                shown = f"`{f.default}`"
            lines.append(f"| `{group}.{f.name}` | {shown} | {(_doc(typ, f.name) or '').replace('|', chr(92) + '|')} |")
    return "\n".join(lines) + "\n"
