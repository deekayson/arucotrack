"""`arucotrack run`: configuration loading, the command line, and whole runs on synthetic clips."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from arucotrack import cli, run
from arucotrack import config as C


def _data_lines(p: Path) -> list[str]:
    return [ln for ln in Path(p).read_text().splitlines() if ln and not ln.startswith("#")]


def _xyz(p: Path) -> dict:
    """(id, frame) -> (x, y, z) of a trajectory file (named columns) or a PeTrack file (columns id frame x y z ...)."""
    lines = _data_lines(p)
    head = ["id", "frame", "x/m", "y/m", "z/m"]
    if lines and lines[0].startswith("id "):
        head, lines = lines[0].split(), lines[1:]
    ix = {n.split("/")[0]: i for i, n in enumerate(head)}
    return {(int(f[ix["id"]]), int(f[ix["frame"]])): (float(f[ix["x"]]), float(f[ix["y"]]), float(f[ix["z"]]))
            for f in (ln.split() for ln in lines)}


def _cfg(tmp_path, text, name="config.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_config_paths_resolve_against_the_config_dir_or_data_root(tmp_path):
    p = _cfg(tmp_path, "video: clips/a.mp4\ncamera: {calibration: cal/calibration.json}\n"
                       "marker: {side_m: 0.18, border_cells: 0.85}\noutput: {work: w, dir: o}\nfloor: {model: plane}\n")
    cfg = C.load(p)
    c = run.resolve(cfg, p)
    assert c["videos"] == [tmp_path / "clips/a.mp4"] and c["calibration"] == tmp_path / "cal/calibration.json"
    assert c["floor"] == "plane" and cfg.marker.border_cells == 0.85 and cfg.gate.mode == "resolve"
    assert cfg.camera.decoder == "auto" and c["work"] == Path("w") and c["out"] == Path("o")
    assert C.det_spec(cfg) == "multipass:subpix ac=3,6,9 scales=1,2" and cfg.detect.patch == 0
    c = run.resolve(C.load(p, ["output.work=" + str(tmp_path / "W"), "output.dir=" + str(tmp_path / "O")]), p,
                     data_root=Path("/data"))
    assert c["videos"] == [Path("/data/clips/a.mp4")] and c["work"] == tmp_path / "W" and c["out"] == tmp_path / "O"
    p2 = _cfg(tmp_path, "video: [a.mp4, /abs/b.mp4]\ncamera: {calibration: c.json}\nmarker: {side_m: 0.18}\n"
                        "floor: {model: surfaces, surfaces: s/ground.json}\ngate: {mode: drop}\n", "b.yaml")
    cfg = C.load(p2)
    c = run.resolve(cfg, p2)
    assert c["videos"] == [tmp_path / "a.mp4", Path("/abs/b.mp4")] and c["floor"] == tmp_path / "s/ground.json"
    assert cfg.gate.mode == "drop" and cfg.marker.border_cells == 0.869


_VC = "video: a.mp4\ncamera: {calibration: c.json}\n"
_OK = _VC + "marker: {side_m: 0.18}\n"


@pytest.mark.parametrize("text, match", [
    ("camera: {calibration: c.json}\n", "video"),
    ("video: a.mp4\n", "calibration"),
    ("video: a.mp4\ncamera: {calibration: c.json}\n", "marker.side_m"),
    (_OK + "floor: {model: stairs}\n", "floor.model"),
    (_OK + "floor: {model: plane, surfaces: s.json}\n", "floor.surfaces"),
    (_OK + "gate: {mode: maybe}\n", "gate.mode"),
    ("video: a.mp4\ncamera: {calibration: c.json}\nmarker: {side_m: 0.18, dictionary: 4x4_50}\n", "dictionary"),
    (_OK + "videos: x\n", "videos"),
    ("video: a.mp4\ncamera: {calibration: c.json}\nmarker: {side_m: 0.18, dictionary: DICT_6X6_FOO}\n", "dictionary"),
    (_VC + "marker: {side_m: 0.18, ids: [-1]}\n", "marker.ids"),
    (_VC + "marker: {side_m: 0.18, ids: [5, 5]}\n", "marker.ids.*unique"),
    (_VC + "marker: {side_m: 0.18, ids: [400]}\n", "marker.ids.*0..249"),
    (_VC + "marker: {side_m: 18}\n", "marker.side_m.*metres"),
    (_VC + "marker: {side_m: 0.001}\n", "marker.side_m.*metres"),
    (_OK + "track: {sigma_acc_m: 0}\n", "track.sigma_acc_m"),
    (_OK + "track: {max_gap_frames: -1}\n", "track.max_gap_frames"),
    (_OK + "track: {interp_cap_frames: -1}\n", "track.interp_cap_frames"),
])
def test_config_errors_are_named(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        C.load(_cfg(tmp_path, text))


def test_the_command_line_has_one_command(capsys):
    assert cli.main(["--help"]) == 0 and "arucotrack run <config.yaml>" in capsys.readouterr().out
    assert cli.main(["--version"]) == 0 and "arucotrack" in capsys.readouterr().out
    for argv in ([], ["track"], ["--bogus"], ["config.yaml"]):
        assert cli.main(argv) == 2
        assert "usage: arucotrack run" in capsys.readouterr().err


def test_run_help_lists_exactly_its_options(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    assert text.startswith("usage: arucotrack run")
    flags = sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", text)))
    assert flags == ["--data-root", "--debug", "--force", "--help", "--out", "--work"]


def test_overrides_may_follow_the_options():
    a = run.parse(["c.yaml", "--data-root", "D", "gate.mode=drop", "output.dir=o", "--force"])
    assert a.overrides == ["gate.mode=drop", "output.dir=o"] and str(a.data_root) == "D" and a.force
    b = run.parse(["c.yaml", "video=[clips/x.mp4]", "--out", "o"])
    assert b.overrides == ["video=[clips/x.mp4]"] and b.out == "o"


def test_data_root_and_config_dir_both_resolve_relative_inputs(tmp_path):
    """`--data-root` wins. Without it the config file's directory is the base (documented in docs/CONFIG.md)."""
    p = _cfg(tmp_path, "video: v.mp4\ncamera: {calibration: cal.json}\nmarker: {side_m: 0.18}\n"
                       "floor: {model: surfaces, surfaces: s.json}\n")
    cfg = C.load(p)
    for root, base in ((None, tmp_path), (Path("/data"), Path("/data"))):
        c = run.resolve(cfg, p, data_root=root)
        assert c["videos"] == [base / "v.mp4"] and c["calibration"] == base / "cal.json"
        assert c["floor"] == base / "s.json"


def test_resolve_refuses_an_unresolved_auto_floor(tmp_path):
    p = _cfg(tmp_path, _OK)
    cfg = C.load(p)
    assert cfg.floor.model == "auto"
    with pytest.raises(ValueError, match="auto"):
        run.resolve(cfg, p)


def _run_clip(tmp_path, video, calib):
    cfg = _cfg(tmp_path, f"video: {video}\ncamera: {{calibration: {calib}}}\nmarker: {{side_m: 0.18}}\n")
    return run.main([str(cfg), "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"),
                      "detect.threads=1"])


def test_run_on_a_video_without_markers_writes_empty_valid_files(tmp_path, blank_video, calib_with_extrinsics):
    from loguru import logger
    msgs: list = []
    sink = logger.add(msgs.append, level="INFO")
    try:
        rc = _run_clip(tmp_path, blank_video, calib_with_extrinsics)
    finally:
        logger.remove(sink)
    assert rc == 0
    for v in ("fused", "fused_interp", "fused_smooth"):
        text = (tmp_path / "out" / f"blank.{v}.txt").read_text()
        lines = text.splitlines()
        assert text.startswith("#") and lines[-1].startswith("id frame t_s x/m") and not [
            ln for ln in lines if ln and not ln.startswith("#") and not ln.startswith("id frame")]
    assert any("no markers detected in" in str(m) for m in msgs)
    out = tmp_path / "out"
    pet = [ln for ln in (out / "blank.petrack.txt").read_text().splitlines() if ln.strip()]
    assert pet and all(ln.startswith("#") for ln in pet), "petrack file: header only"
    side = json.loads((out / "blank.sidecar.json").read_text())
    assert side["stem"] == "blank" and side["orientation"]["rows"] == 0


def test_run_without_extrinsics_runs_with_the_default_floor(tmp_path, blank_video, calib_intrinsics_only):
    assert _run_clip(tmp_path, blank_video, calib_intrinsics_only) == 0
    assert "CAMERA frame" in (tmp_path / "out" / "blank.fused.txt").read_text()


def test_run_run_leaves_the_callers_opencv_thread_count_unchanged(tmp_path, blank_video, calib_with_extrinsics):
    import cv2
    cv2.setNumThreads(3)
    assert _run_clip(tmp_path, blank_video, calib_with_extrinsics) == 0          # detect.threads=1 inside
    assert cv2.getNumThreads() == 3


def test_run_refuses_a_fisheye_calibration_before_any_stage(tmp_path, blank_video, capsys):
    """`run` detects on raw frames and poses with the pinhole model: a fisheye calibration is refused before
    detection, so no cache is written."""
    calib = tmp_path / "fisheye.json"
    calib.write_text(json.dumps({"camera_id": "cam_fisheye", "model": "OPENCV_FISHEYE",
                                 "image_size": [320, 240],
                                 "intrinsics": {"fx": 120.0, "fy": 120.0, "cx": 160.0, "cy": 120.0,
                                                "dist": [0.1, -0.01, 0.0, 0.0]}}))
    assert _run_clip(tmp_path, blank_video, calib) != 0
    err = " ".join(capsys.readouterr().err.split())
    assert "fisheye calibration" in err and "needs a pinhole calibration" in err
    assert not (tmp_path / "work").exists() or not any((tmp_path / "work").iterdir())
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def _clip_cfg(tmp_path, video, calib, extra=""):
    return _cfg(tmp_path, f"video: {video}\ncamera: {{calibration: {calib}}}\n"
                          f"marker: {{side_m: 0.5, border_cells: 1.0}}\n{extra}", "e2e.yaml")


def _run(tmp_path, cfg, *more):
    return run.main([str(cfg), "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work"), "detect.threads=1",
                      *more])


def test_run_end_to_end_on_a_rendered_marker_writes_rows(tmp_path, marker_video, calib_with_extrinsics):
    """detect -> refine -> pose -> files on a synthetic clip with one marker 1.7 m above the floor plane."""
    from tests.conftest import E2E_ID, E2E_Z_M
    assert _run(tmp_path, _clip_cfg(tmp_path, marker_video, calib_with_extrinsics)) == 0
    out = tmp_path / "out"
    for v in ("fused", "fused_interp", "fused_smooth"):
        rows = _xyz(out / f"walk.{v}.txt")
        assert sorted(rows) == [(E2E_ID, k) for k in range(8)], v
        assert all(abs(z - E2E_Z_M) < 0.02 for _, _, z in rows.values()), v
    assert len(_xyz(out / "walk.petrack.txt")) == 8
    side = json.loads((out / "walk.sidecar.json").read_text())
    assert side["detection"]["frames_with_marker"] == 8 and side["frame"] == "world"


def test_stage_stamps_name_the_code_and_a_changed_code_state_recomputes(tmp_path, marker_video, calib_with_extrinsics):
    import arucotrack
    cfg = _clip_cfg(tmp_path, marker_video, calib_with_extrinsics)
    assert _run(tmp_path, cfg) == 0
    stamps = sorted((tmp_path / "work").glob("*.npz.run.json"))
    assert len(stamps) == 2
    for p in stamps:
        st = json.loads(p.read_text())["stamp"]
        assert st["code"]["version"] == arucotrack.__version__ and "commit" in st["code"]
    det = next(p for p in stamps if ".refined." not in p.name)
    d = json.loads(det.read_text())
    d["stamp"]["code"]["version"] = "0.0.0"
    det.write_text(json.dumps(d))
    assert _run(tmp_path, cfg) == 0
    assert json.loads(det.read_text())["stamp"]["code"]["version"] == arucotrack.__version__


def _calib_size(tmp_path, size):
    from tests.conftest import _calib_doc
    d = _calib_doc(True)
    d["image_size"] = size
    p = tmp_path / "calib_size.json"
    p.write_text(json.dumps(d))
    return p


def test_run_refuses_a_video_whose_size_differs_from_the_calibration(tmp_path, blank_video, capsys):
    assert _run_clip(tmp_path, blank_video, _calib_size(tmp_path, [640, 480])) == 2
    err = " ".join(capsys.readouterr().err.split())
    assert "320x240" in err and "640x480" in err and "blank.mp4" in err
    assert not (tmp_path / "work").exists() or not list((tmp_path / "work").glob("*.npz"))


def test_refine_refuses_a_video_whose_size_differs_from_the_calibration(tmp_path, blank_video):
    from arucotrack.detection import detect
    from arucotrack.refinement import refine_cache
    from arucotrack.calibration.camera import Calib
    s = detect.run(blank_video, cache_root=tmp_path, source="sw_y", det_spec="multipass:subpix ac=3 scales=1",
                   tag="t", stem="blank")
    calib = Calib.from_json(_calib_size(tmp_path, [640, 480]), marker_m=0.18)
    with pytest.raises(ValueError, match="320x240.*640x480"):
        refine_cache.run(s["cache"], calib=calib, video=blank_video, out=tmp_path / "r.refined.npz",
                         border_cells=0.869, source="sw_y")


def test_run_refuses_two_videos_with_one_stem(tmp_path, blank_video, calib_with_extrinsics, capsys):
    import shutil
    (tmp_path / "b").mkdir()
    shutil.copy(blank_video, tmp_path / "b" / "blank.mp4")
    cfg = _cfg(tmp_path, f"video: [{blank_video}, {tmp_path / 'b' / 'blank.mp4'}]\n"
                         f"camera: {{calibration: {calib_with_extrinsics}}}\nmarker: {{side_m: 0.18}}\n")
    assert run.main([str(cfg), "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")]) == 2
    assert "blank" in capsys.readouterr().err and not (tmp_path / "work").exists()


def test_run_refuses_another_video_of_the_same_stem_in_a_used_work_dir(tmp_path, blank_video, calib_with_extrinsics,
                                                                        capsys):
    import shutil
    assert _run_clip(tmp_path, blank_video, calib_with_extrinsics) == 0
    (tmp_path / "b").mkdir()
    other = tmp_path / "b" / "blank.mp4"
    shutil.copy(blank_video, other)
    capsys.readouterr()
    assert _run_clip(tmp_path, other, calib_with_extrinsics) == 2
    err = " ".join(capsys.readouterr().err.split())
    assert str(blank_video) in err and str(other) in err and "--work" in err


@pytest.mark.parametrize("doc, match", [
    (None, "does not exist"),
    ({"frame": "world_m"}, "surfaces"),
    ({"frame": "world_m", "surfaces": [{"id": 3, "kind": "t", "polygon_xy": [[0, 0], [1, 0]],
                                        "plane": [0, 0, 1, 0]}]}, "surface 3.*three"),
    ({"frame": "world_m", "surfaces": [{"id": 4, "kind": "t", "polygon_xy": [[0, 0], [1, 0], [1, 1]],
                                        "plane": [1, 0, 0, 0]}]}, "surface 4.*c"),
    ({"frame": "world_m", "surfaces": [{"id": 5, "kind": "t", "polygon_xy": [[0, 0], [1, 0], [1, 1]]}]},
     "surface 5.*plane"),
])
def test_run_checks_the_walking_surfaces_before_detection(tmp_path, blank_video, calib_with_extrinsics, capsys,
                                                           doc, match):
    surf = tmp_path / "surfaces.json"
    if doc is not None:
        surf.write_text(json.dumps(doc))
    cfg = _cfg(tmp_path, f"video: {blank_video}\ncamera: {{calibration: {calib_with_extrinsics}}}\n"
                         f"marker: {{side_m: 0.18}}\nfloor: {{model: surfaces, surfaces: {surf}}}\n")
    assert run.main([str(cfg), "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")]) == 2
    err = " ".join(capsys.readouterr().err.split())
    import re
    assert re.search(match, err) and "surfaces.json" in err
    assert not (tmp_path / "work").exists() or not list((tmp_path / "work").glob("*.npz"))


def test_run_reports_an_unexpected_error_without_a_traceback(tmp_path, blank_video, calib_with_extrinsics, capsys,
                                                             monkeypatch):
    from arucotrack.detection import detect

    def boom(*a, **k):
        raise RuntimeError("ffmpeg exited 1: broken pipe")

    monkeypatch.setattr(detect, "run", boom)
    assert _run_clip(tmp_path, blank_video, calib_with_extrinsics) == 1
    err = capsys.readouterr().err
    assert "arucotrack run: RuntimeError: ffmpeg exited 1: broken pipe" in err and "Traceback" not in err


def test_zero_detections_skip_the_refine_decode(tmp_path, blank_video, calib_with_extrinsics, monkeypatch):
    from arucotrack.refinement import refine_cache

    def no_decode(*a, **k):
        raise AssertionError("the video was decoded for refinement although nothing was detected")

    monkeypatch.setattr(refine_cache, "frames", no_decode)
    assert _run_clip(tmp_path, blank_video, calib_with_extrinsics) == 0
    assert list((tmp_path / "work").glob("*.resolve.refined.npz"))


def test_refinement_errors_on_most_rows_stop_the_run(tmp_path, marker_video, calib_with_extrinsics, capsys,
                                                     monkeypatch):
    from arucotrack.refinement import refine

    def broken(*a, **k):
        raise TypeError("an incompatible library")

    monkeypatch.setattr(refine, "refine", broken)
    assert _run(tmp_path, _clip_cfg(tmp_path, marker_video, calib_with_extrinsics)) == 1
    err = capsys.readouterr().err
    assert "8 of 8" in err and "TypeError: an incompatible library" in err
    assert not list((tmp_path / "out").glob("*.fused.txt"))


def test_a_one_frame_video_is_refused_with_a_clear_message(tmp_path, calib_with_extrinsics, capsys):
    from tests.conftest import write_gray_video
    vid = write_gray_video(tmp_path / "one.mp4", [np.full((240, 320), 128, np.uint8)])
    assert _run_clip(tmp_path, vid, calib_with_extrinsics) == 2
    assert "at least two frames" in capsys.readouterr().err


def test_an_interrupted_rewrite_is_redone_on_the_next_run(tmp_path, marker_video, calib_with_extrinsics,
                                                          monkeypatch):
    from arucotrack.io import petrack_export
    cfg = _clip_cfg(tmp_path, marker_video, calib_with_extrinsics)
    assert _run(tmp_path, cfg) == 0
    good = _data_lines(tmp_path / "out" / "walk.fused_smooth.txt")
    real = petrack_export.write

    def killed(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(petrack_export, "write", killed)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, cfg, "--force")
    assert not (tmp_path / "work" / "walk.trajectories.run.json").exists()
    assert not list((tmp_path / "out").glob("*.tmp"))
    monkeypatch.setattr(petrack_export, "write", real)
    assert _run(tmp_path, cfg) == 0
    assert _data_lines(tmp_path / "out" / "walk.fused_smooth.txt") == good
    assert (tmp_path / "work" / "walk.trajectories.run.json").exists()


def test_the_default_border_constant_is_logged_as_the_default(tmp_path, marker_video, calib_with_extrinsics):
    from loguru import logger
    msgs: list = []
    sink = logger.add(msgs.append, level="INFO")
    try:
        cfg = _cfg(tmp_path, f"video: {marker_video}\ncamera: {{calibration: {calib_with_extrinsics}}}\n"
                             f"marker: {{side_m: 0.5}}\n")
        assert _run(tmp_path, cfg) == 0
    finally:
        logger.remove(sink)
    text = " ".join(str(m) for m in msgs)
    assert "border_cells 0.869 (default)" in text and "(override)" not in text


def test_camera_frame_output_measures_the_view_angle_from_the_optical_axis(tmp_path, marker_video,
                                                                          calib_intrinsics_only):
    """Without a camera pose `view_basler_deg` is the angle between the viewing ray and +z, small near the image
    centre, and the PeTrack header names the camera frame."""
    assert _run(tmp_path, _clip_cfg(tmp_path, marker_video, calib_intrinsics_only)) == 0
    lines = (tmp_path / "out" / "walk.fused.txt").read_text().splitlines()
    head = next(ln for ln in lines if ln.startswith("id "))
    k = head.split().index("view_basler_deg")
    ang = [float(ln.split()[k]) for ln in lines if ln and not ln.startswith(("#", "id "))]
    assert ang and all(0 <= a < 20 for a in ang)
    pet = (tmp_path / "out" / "walk.petrack.txt").read_text().splitlines()
    assert sum(ln.startswith("# framerate:") for ln in pet) == 1
    assert not any("z up" in ln for ln in pet if ln.startswith("#"))
    assert any(ln.startswith("# metres, CAMERA frame") and "optical axis" in ln for ln in pet)


def test_world_frame_petrack_header_keeps_one_framerate_line(tmp_path, marker_video, calib_with_extrinsics):
    assert _run(tmp_path, _clip_cfg(tmp_path, marker_video, calib_with_extrinsics)) == 0
    pet = (tmp_path / "out" / "walk.petrack.txt").read_text().splitlines()
    assert sum(ln.startswith("# framerate:") for ln in pet) == 1
    assert any(ln.startswith("# metres, world frame") and "z up" in ln for ln in pet)
