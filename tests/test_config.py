"""The run configuration (`arucotrack.config`): defaults equal the code's constants, overrides, refusals, docs."""
import inspect
from pathlib import Path

import numpy as np
import pytest

from omegaconf import OmegaConf

from arucotrack import config as C
from arucotrack import pipeline
from arucotrack.calibration.camera import Calib
from arucotrack.detection import detectors
from arucotrack.refinement import refine
from arucotrack.refinement import refine_cache

ROOT = Path(__file__).resolve().parents[1]


def _defaults(fn):
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect._empty}


def test_the_default_configuration_is_the_code_s_constants():
    """Each default in the schema equals the value the code uses."""
    d = C.default()
    # marker
    assert d.marker.dictionary == detectors.dictionary_name(detectors.DICT) == "DICT_ARUCO_MIP_36h12"
    assert OmegaConf.is_missing(d.marker, "side_m")       # the printed size is the user's to state
    assert d.marker.border_cells == refine.BORDER_CELLS
    assert d.marker.ids is None
    # detect
    mp = detectors.MultiPass()
    assert list(d.detect.constants) == list(mp.constants)
    assert list(d.detect.scales) == [float(s) for s in mp.scales]
    assert (d.detect.win_size_max, d.detect.win_size_step, d.detect.pixels_per_cell) == (
        mp.params["adaptiveThreshWinSizeMax"], mp.params["adaptiveThreshWinSizeStep"],
        mp.params["perspectiveRemovePixelPerCell"])
    assert d.detect.corner_refinement == mp.refine and d.detect.iou == mp.iou
    assert (d.detect.periphery_rule, d.detect.theta_max_deg) == ("any_corner", 65.0)     # fisheye only, not used
    # refine
    rd = _defaults(refine.refine)
    assert (d.refine.samples_per_edge, d.refine.iterations, d.refine.cauchy_c) == (rd["per_edge"], rd["iters"],
                                                                                   rd["cauchy_c"])
    assert d.refine.inset_cells == _defaults(refine.interior_samples)["inset"]
    assert d.refine.profile_step_px == refine.PROFILE_STEP == _defaults(refine._edge_offsets)["step"]
    assert d.refine.min_contrast == _defaults(refine._edge_offsets)["min_contrast"]
    assert d.refine.gauss_newton_steps == _defaults(refine._fit_homography_to_lines)["steps"]
    side = lambda s: np.array([[0, 0], [s, 0], [s, s], [0, s]], float)
    assert refine._half_window(side(60.0)) == pytest.approx(0.42 * 60 / refine.CELLS)
    assert refine._half_window(side(8.0)) == 1.0 and refine._half_window(side(800.0)) == 4.0
    src = inspect.getsource(refine.refine)
    assert f"used < {d.refine.min_samples}" in src
    assert d.refine.max_move_cells == 1.0 and "shift > cell_px" in src
    assert d.refine.crop_px == refine_cache.CROP
    assert d.refine.velocity_window_frames == refine_cache.MAX_NEIGHBOUR_GAP
    rc = _defaults(refine_cache.run)
    assert d.refine.blur_window == rc["blur_window"] and d.refine.sigma_optics is None
    assert (rc["samples_per_edge"], rc["iterations"], rc["cauchy_c"]) == (3, 3, 1.5)
    # gate
    g = d.gate
    assert g.mode == rc["gate_mode"] and g.mode in refine_cache.GATE_MODES and C.GATE_MODES == refine_cache.GATE_MODES
    assert (g.max_edge_rms_px, g.min_shift_px, g.max_shift_frac, g.max_downweighted_frac, g.agree_px,
            g.window_frames) == (refine_cache.MAX_EDGE_RMS_PX, refine_cache.MIN_SHIFT_PX, refine_cache.MAX_SHIFT_FRAC,
                                 refine_cache.MAX_DOWNWEIGHTED_FRAC, refine_cache.GATE_AGREE_PX,
                                 refine_cache.GATE_WINDOW_FRAMES)
    # track
    assert pipeline.Params.from_config(d) == pipeline.Params()
    assert d.track.sigma_acc_m == pipeline.SIGMA_ACC_M
    assert (pipeline.MIN_TRIPLES, pipeline.SIGMA_S_FALLBACK_M, pipeline.SIGMA_S_FLOOR_M) == (20, 0.002, 1e-4)
    # detector spec and cache tag derived from the defaults: a cache made with the defaults must stay valid
    cfg = C.load(None, ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18"])
    assert C.det_spec(cfg) == "multipass:subpix ac=3,6,9 scales=1,2"
    assert C.det_tag(cfg) == "multipass_ac369_s12"
    spec = detectors.parse_spec(C.det_spec(cfg))
    assert spec.constants == mp.constants and spec.params == mp.params and spec.iou == mp.iou


def test_dotted_overrides_merge_over_the_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("video: a.mp4\ncamera: {calibration: c.json}\nmarker: {side_m: 0.18}\ngate: {mode: resolve}\n")
    cfg = C.load(p, ["gate.mode=drop", "camera.frames=120", "marker.ids=[17,18]", "detect.constants=[3,6]",
                     "track.height_band_m=null"])
    assert cfg.gate.mode == "drop" and cfg.camera.frames == 120 and list(cfg.marker.ids) == [17, 18]
    assert list(cfg.video) == ["a.mp4"] and cfg.track.height_band_m is None
    assert C.det_spec(cfg) == "multipass:subpix ac=3,6 scales=1,2 roster"
    assert C.det_tag(cfg) == "multipass_ac36_s12_roster"
    assert "gate drop" in C.summary(cfg) and C.to_yaml(cfg) != C.to_yaml(C.load(p))


@pytest.mark.parametrize("override, match", [
    (["gate.mode=sometimes"], "gate.mode"),
    (["gate.mode=legacy"], "gate.mode"),
    (["marker.dictionary=DICT_4X4_50"], "6x6"),
    (["marker.dictionary=MIP"], "predefined"),
    (["refine.inset_cells=0.3"], "fixed"),
    (["detect.theta_max_deg=70"], "fixed"),
    (["camera.decoder=gpu"], "camera.decoder"),
    (["detect.threads=-1"], "detect.threads"),
    (["refine.blur_window=true"], "sigma_optics"),
    (["track.z_range_m=[2,1]"], "z_range_m"),
    (["marker.side_m=abc"], "side_m"),
    (["nonsense.key=1"], "nonsense"),
])
def test_bad_values_are_refused_by_name(override, match):
    with pytest.raises(ValueError, match=match):
        C.load(None, ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18", *override])


def test_the_marker_side_is_required():
    with pytest.raises(ValueError, match="marker.side_m"):
        C.load(None, ["video=[a.mp4]", "camera.calibration=c.json"])


def test_a_6x6_dictionary_other_than_the_default_is_accepted():
    cfg = C.load(None, ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18", "marker.dictionary=DICT_6X6_250"])
    assert C.det_tag(cfg).endswith("_6x6_250")
    det = detectors.parse_spec(C.det_spec(cfg), dictionary=detectors.dictionary_id(cfg.marker.dictionary))
    assert det.dictionary == detectors.dictionary_id("DICT_6X6_250")


def test_without_extrinsics_the_floor_and_height_gates_are_refused():
    cfg = C.load(None, ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18"])
    with pytest.raises(ValueError, match=r"floor.model=none track.z_range_m=null track.height_band_m=null"):
        C.check(cfg, has_extrinsics=False)
    ok = C.load(None, ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18", "floor.model=none", "track.z_range_m=null",
                       "track.height_band_m=null"])
    C.check(ok, has_extrinsics=False)
    C.check(cfg, has_extrinsics=True)


def test_every_field_is_documented_and_docs_config_md_is_current():
    md = C.markdown()
    for group, typ in C._groups():
        for f in __import__("dataclasses").fields(typ):
            assert C._doc(typ, f.name), f"{group}.{f.name} has no docstring"
            assert f"`{group}.{f.name}`" in md
    doc = ROOT / "docs" / "CONFIG.md"
    assert doc.read_text() == md, ("docs/CONFIG.md is stale: python -c \"from arucotrack import config; "
                                  "print(config.markdown(), end='')\" > docs/CONFIG.md")


def test_the_example_config_is_an_instance_of_the_schema():
    cfg = C.load(ROOT / "config.example.yaml")
    assert cfg.camera.calibration and cfg.video


def test_track_sigma_s_fixes_the_measurement_noise_and_the_default_estimates_it():
    base = ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18"]
    d = C.load(None, base)
    assert d.track.sigma_s == C.TrackConfig.sigma_s and pipeline.Params.from_config(d).sigma_s_m is None
    assert "sigma_s" not in C.summary(d)                       # the default adds nothing to the header line
    for given in ("0.0023", "'0.0023'"):
        cfg = C.load(None, [*base, f"track.sigma_s={given}"])
        assert pipeline.Params.from_config(cfg).sigma_s_m == 0.0023
        assert C.summary(cfg).endswith("sigma_s 0.0023")


@pytest.mark.parametrize("value", ["abc", "0", "-0.001", "0.2"])
def test_track_sigma_s_refuses_what_is_not_a_noise_in_metres(value):
    with pytest.raises(ValueError, match="track.sigma_s"):
        C.load(None, ["video=[a.mp4]", "camera.calibration=c.json", "marker.side_m=0.18", f"track.sigma_s={value}"])


def _floor_cfg(calib_path, **extra):
    return C.load(None, ["video=v.mp4", f"camera.calibration={calib_path}", "marker.side_m=0.18",
                         *[f"{k}={v}" for k, v in extra.items()]])


def test_floor_auto_is_plane_with_extrinsics(calib_with_extrinsics):
    cfg = _floor_cfg(calib_with_extrinsics)
    assert cfg.floor.model == "auto"
    r = C.resolve_auto_floor(cfg, Calib.from_json(calib_with_extrinsics, marker_m=0.18))
    assert r.floor.model == "plane" and r.track.z_range_m == cfg.track.z_range_m
    assert r.track.height_band_m == cfg.track.height_band_m


def test_floor_auto_is_none_without_extrinsics_and_gates_off(calib_intrinsics_only):
    cfg = _floor_cfg(calib_intrinsics_only)
    r = C.resolve_auto_floor(cfg, Calib.from_json(calib_intrinsics_only, marker_m=0.18))
    assert r.floor.model == "none" and r.track.z_range_m is None and r.track.height_band_m is None
    C.check(r, has_extrinsics=False)


def test_explicit_plane_without_extrinsics_is_an_error(calib_intrinsics_only):
    cfg = _floor_cfg(calib_intrinsics_only, **{"floor.model": "plane"})
    calib = Calib.from_json(calib_intrinsics_only, marker_m=0.18)
    with pytest.raises(ValueError, match="no extrinsics"):
        C.check(C.resolve_auto_floor(cfg, calib), has_extrinsics=False)


def test_explicit_settings_are_not_touched_by_resolve(calib_with_extrinsics, calib_intrinsics_only):
    cfg = _floor_cfg(calib_with_extrinsics, **{"floor.model": "none", "track.z_range_m": "null"})
    r = C.resolve_auto_floor(cfg, Calib.from_json(calib_with_extrinsics, marker_m=0.18))
    assert r.floor.model == "none" and r.track.height_band_m == cfg.track.height_band_m


def test_example_config_loads_with_only_the_documented_fields():
    cfg = C.load(ROOT / "config.example.yaml", ["video=x.mp4", "camera.calibration=c.json"])
    assert cfg.marker.side_m > 0 and cfg.floor.model == "auto"
    keys = {k for k, v in OmegaConf.to_container(OmegaConf.load(ROOT / "config.example.yaml")).items()}
    assert keys <= {"video", "camera", "marker"}


def test_the_acceleration_prior_default_is_the_measured_value_of_the_paper():
    """0.45 mm per frame squared (the paper, Appendix A), in the configuration and in the pipeline's own default."""
    assert C.default().track.sigma_acc_m == 0.00045
    assert pipeline.SIGMA_ACC_M == pipeline.Params().sigma_acc_m == 0.00045
