"""The single-camera pipeline (`arucotrack.pipeline`): row rule, floor modes, gates, files, idempotence."""
import json

import numpy as np
import pytest

from arucotrack.calibration.camera import Calib
from arucotrack import pipeline as P
from arucotrack.io import trajwrite as TW
from tests.conftest import MARKER_M, calib_doc, make_fisheye_calib, read_trajectory
from tests.conftest import marker_quad as _quad


def _calib(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps(calib_doc("cam_overhead")))
    return Calib.from_json(p, marker_m=MARKER_M)


def _walk(n=120, z=1.7, gap=(50, 60)):
    """Two ids walking along y at head height z, with a detection gap. Frames 0..n-1."""
    rows = []
    for mid, x0 in ((17, 0.2), (23, -0.4)):
        for f in range(n):
            if gap[0] <= f < gap[1] and mid == 17:
                continue
            rows.append((mid, f, np.array([x0 + 0.002 * np.sin(f / 7), -2.0 + 0.012 * f, z + 0.01 * np.sin(f / 5)])))
    return rows


def _cache(path, calib, rows, *, resolution=None, used=None, n_frames=None):
    seq = np.array([r[1] for r in rows]); ids = np.array([r[0] for r in rows], np.int32)
    cor = np.array([calib.project(_quad(r[2])) for r in rows], np.float32)
    n_frames = n_frames or int(seq.max()) + 1
    prov = {"input": {"relpath": "clip.mp4"}, "corner_convention": "centre",
            "environment": {"decoder": "x", "pixel_pipeline": "y"}, "parameters": {}}
    arrs = dict(det_seq=seq, det_pts=seq / 60.0, det_id=ids, det_corners=cor, frame_pts_all=np.arange(n_frames) / 60.0,
                det_refine_used=np.ones(len(seq), bool) if used is None else np.asarray(used, bool),
                provenance=np.array(json.dumps(prov)))
    if resolution is not None:
        arrs["det_gate_resolution"] = np.asarray(resolution, np.uint8)
    np.savez(path, **arrs)
    return path


def test_row_rule_keeps_resolutions_1_to_3_else_det_refine_used():
    keep, rule = P.row_rule({"det_gate_resolution": np.array([0, 1, 2, 3, 4]), "det_refine_used": np.ones(5, bool)})
    assert keep.tolist() == [False, True, True, True, False] and "4 unverified dropped" in rule
    keep, rule = P.row_rule({"det_refine_used": np.array([True, False, True])})
    assert keep.tolist() == [True, False, True] and "det_refine_used" in rule


def test_flat_floor_covers_everything_at_z0():
    xyz = np.array([[0.0, 0.0, 1.7], [-900.0, 800.0, 0.5], [3.0, -2.0, -0.1]])
    xf, sid = P.S.project_to_surfaces(xyz, P.flat_floor())
    assert sid.tolist() == [0, 0, -1]                       # a head below the plane has no floor under it
    assert np.allclose(xf[:2, 2], 0.0) and np.allclose(xf[:2, :2], xyz[:2, :2])


def _run(tmp_path, calib, rows, floor, name, params=P.Params(), **kw):
    c = _cache(tmp_path / f"{name}.npz", calib, rows, **kw)
    out = tmp_path / name
    r = P.metric_step("clip", c, c, out, tmp_path / "work", calib=calib, floor=floor, params=params,
                      camera="cam_overhead")
    return r, out


def test_a_flat_plane_run_equals_a_no_surface_run_on_x_y_z(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk()                                            # every head inside the 0.3-2.8 m band
    _, a = _run(tmp_path, calib, rows, None, "none")
    _, b = _run(tmp_path, calib, rows, P.FLAT, "plane")
    for v in ("fused", "fused_interp", "fused_smooth"):
        A, B = read_trajectory(a / f"clip.{v}.txt"), read_trajectory(b / f"clip.{v}.txt")
        assert len(A["id"]) == len(B["id"]) > 0
        for col in ("id", "frame", "x", "y", "z"):
            assert np.array_equal(A[col], B[col]), (v, col)
        assert np.isnan(A["z_floor"]).all() and (A["surface_id"] == -1).all()
        assert np.allclose(B["z_floor"], 0.0) and (B["surface_id"] == 0).all() and (B["floor_src"] == 1).all()
    assert np.allclose(read_trajectory(b / "clip.fused.txt")["z"], 1.7, atol=0.02)


def test_the_plane_turns_the_height_band_on(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk(z=1.7) + [(31, f, np.array([1.0, 0.0, 0.2])) for f in range(40)]     # a marker 0.2 m above the floor
    _, a = _run(tmp_path, calib, rows, None, "none")
    _, b = _run(tmp_path, calib, rows, P.FLAT, "plane")
    assert 31 in read_trajectory(a / "clip.fused.txt")["id"]
    assert 31 not in read_trajectory(b / "clip.fused.txt")["id"]
    side = json.loads((b / "clip.sidecar.json").read_text())
    assert side["gating"]["height_gated_raw"] == 40 and side["floor"]["mode"] == "plane"


def test_unverified_rows_are_not_posed_and_gaps_fill_and_smooth(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk()
    res = np.ones(len(rows), np.uint8)
    res[5] = 4                                                # one unverified row of id 17 (frame 5)
    _, out = _run(tmp_path, calib, rows, P.FLAT, "res", resolution=res)
    raw = read_trajectory(out / "clip.fused.txt")
    assert not ((raw["id"] == 17) & (raw["frame"] == 5)).any()
    it = read_trajectory(out / "clip.fused_interp.txt")
    k = (it["id"] == 17) & (it["frame"] == 55)
    assert k.sum() == 1 and it["observed"][k][0] == 0 and it["flags"][k][0] == "interp"
    sm = read_trajectory(out / "clip.fused_smooth.txt")
    assert set(sm["frame"][sm["id"] == 17]) == set(range(120))     # gap 10 <= max_gap: one segment, every frame
    side = json.loads((out / "clip.sidecar.json").read_text())
    assert "4 unverified dropped" in side["row_rule"] and side["gating"]["dropped_unrefined"] == 1


def test_corner_column_names_the_corners_each_row_was_posed_from(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk()
    res = np.ones(len(rows), np.uint8)
    res[10], res[11] = 2, 3                                   # id 17, frames 10 and 11: detector, refused refinement
    _, out = _run(tmp_path, calib, rows, P.FLAT, "res", resolution=res)
    raw = read_trajectory(out / "clip.fused.txt")
    src = {int(f): s for i, f, s in zip(raw["id"], raw["frame"], raw["corner"]) if i == 17}
    assert src[10] == "detector" and src[11] == "refused_refinement" and src[12] == "refined"
    assert set(raw["corner"]) == {"refined", "detector", "refused_refinement"}
    _, drop = _run(tmp_path, calib, rows, P.FLAT, "drop")          # no resolutions (drop mode): accepted refinements
    assert set(read_trajectory(drop / "clip.fused.txt")["corner"]) == {"refined"}


def test_headers_name_the_calibration_file_without_its_path(tmp_path):
    calib = _calib(tmp_path)
    _, out = _run(tmp_path, calib, _walk(), P.FLAT, "hdr")
    line = next(s for s in (out / "clip.fused.txt").read_text().splitlines() if s.startswith("# calibration:"))
    assert line == "# calibration: calibration.json" and str(tmp_path) not in line
    side = json.loads((out / "clip.sidecar.json").read_text())
    assert side["calibration"]["path"] == "calibration.json" and str(tmp_path) not in json.dumps(side)


def test_out_files_carry_no_hash_commit_or_machine_path(tmp_path):
    """The headers, the sidecar and the observed file name what made them in words: no sha256, no git commit, no
    absolute path. The key that lets an unchanged re-run skip the step lives in the work folder."""
    import re
    calib = _calib(tmp_path)
    _, out = _run(tmp_path, calib, _walk(), P.FLAT, "plain")
    work = tmp_path / "work"
    hex64 = re.compile(r"\b[0-9a-f]{40,64}\b")
    texts = {f: (out / f"clip.{f}").read_text() for f in ("fused.txt", "fused_interp.txt", "fused_smooth.txt",
                                                          "petrack.txt", "sidecar.json", "petrack.observed.json")}
    for name, text in texts.items():
        head = "\n".join(ln for ln in text.splitlines() if ln.startswith("#")) if name.endswith(".txt") else text
        assert "sha256" not in head and "git" not in head and not hex64.search(head), name
        assert str(tmp_path) not in head, name
    side = json.loads(texts["sidecar.json"])

    def strings(o):
        if isinstance(o, dict):
            return [x for v in o.values() for x in strings(v)]
        return [x for v in o for x in strings(v)] if isinstance(o, list) else [o] if isinstance(o, str) else []
    assert not [v for v in strings(side) if v.startswith("/")], "no absolute path in the sidecar"
    assert side["code"].startswith("arucotrack ") and "inputs_key" not in side and "dataset" not in side
    assert json.loads((work / "clip.trajectories.run.json").read_text())["inputs_key"]


def test_short_segments_are_dropped_from_the_smoothed_set_only(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk() + [(40, f, np.array([0.5, 0.5, 1.6])) for f in (10, 11, 12)]        # 3 rows < min_segment 5
    _, out = _run(tmp_path, calib, rows, P.FLAT, "short")
    assert 40 in read_trajectory(out / "clip.fused.txt")["id"]
    assert 40 not in read_trajectory(out / "clip.fused_smooth.txt")["id"]


def test_metric_step_is_idempotent_and_writes_every_file(tmp_path):
    calib = _calib(tmp_path)
    c = _cache(tmp_path / "c.npz", calib, _walk())
    kw = dict(calib=calib, floor=P.FLAT, camera="cam_overhead")
    r1 = P.metric_step("clip", c, c, tmp_path / "o", tmp_path / "w", **kw)
    assert not r1["skipped"]
    for suf in ("fused.txt", "fused_interp.txt", "fused_smooth.txt", "petrack.txt", "petrack.observed.json",
                "orient.npz", "sidecar.json"):
        assert (tmp_path / "o" / f"clip.{suf}").is_file(), suf
    assert P.metric_step("clip", c, c, tmp_path / "o", tmp_path / "w", **kw)["skipped"]
    assert not P.metric_step("clip", c, c, tmp_path / "o", tmp_path / "w", force=True, **kw)["skipped"]
    head = (tmp_path / "o" / "clip.fused.txt").read_text().splitlines()[0]
    assert head.startswith("# arucotrack 0.1.0 trajectories | clip clip | fused: measured rows")


def test_skip_is_per_output_folder(tmp_path):
    # two output folders sharing one work folder: a configuration run into folder b must not let folder a,
    # written with another configuration, pass as up to date
    calib = _calib(tmp_path)
    c = _cache(tmp_path / "c.npz", calib, _walk())
    kw = dict(calib=calib, floor=P.FLAT, camera="cam_overhead")
    w = tmp_path / "w"
    assert not P.metric_step("clip", c, c, tmp_path / "a", w, config_line="one", **kw)["skipped"]
    assert not P.metric_step("clip", c, c, tmp_path / "b", w, config_line="two", **kw)["skipped"]
    assert not P.metric_step("clip", c, c, tmp_path / "a", w, config_line="two", **kw)["skipped"]
    assert P.metric_step("clip", c, c, tmp_path / "a", w, config_line="two", **kw)["skipped"]


def test_writer_formats_deterministically(tmp_path):
    assert TW.num(-0.00001, 4) == "0.0000" and TW.num(float("nan"), 3) == "nan" and TW.num(1.23456, 2) == "1.23"
    c = P.empty_cols()
    with pytest.raises(KeyError):
        TW.write_interface(tmp_path / "x.txt", {"id": []}, [])
    TW.write_interface(tmp_path / "e.txt", c, ["h"])
    assert (tmp_path / "e.txt").read_text().splitlines() == ["# h", TW.HEADER]


def _smooth_xyz(out):
    s = read_trajectory(out / "clip.fused_smooth.txt")
    return np.c_[s["x"], s["y"], s["z"]]


def test_a_fixed_sigma_s_replaces_the_estimate_and_is_recorded(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk()
    _, est = _run(tmp_path, calib, rows, P.FLAT, "est")
    side = json.loads((est / "clip.sidecar.json").read_text())
    s_est = side["sigma_s"]["1"]["sigma_m"]
    assert side["sigma_s"]["1"]["origin"] == "source" and side["parameters"]["sigma_s_m"] is None
    head = [l for l in (est / "clip.fused_smooth.txt").read_text().splitlines() if l.startswith("# track:")][0]
    assert "sigma_s" not in head and "sigma_acc 0.45 mm/frame^2" in head
    # the estimate given as the fixed value gives the same rows. Another value gives other rows
    _, same = _run(tmp_path, calib, rows, P.FLAT, "same", params=P.Params(sigma_s_m=s_est))
    assert np.array_equal(_smooth_xyz(same), _smooth_xyz(est))
    _, fix = _run(tmp_path, calib, rows, P.FLAT, "fix", params=P.Params(sigma_s_m=0.0023))
    assert not np.allclose(_smooth_xyz(fix), _smooth_xyz(est))
    side = json.loads((fix / "clip.sidecar.json").read_text())
    assert side["sigma_s"]["1"]["sigma_m"] == 0.0023 and side["sigma_s"]["1"]["origin"] == "parameter"
    assert side["sigma_s"]["1"]["n_triples"] > 0 and side["parameters"]["sigma_s_m"] == 0.0023
    for v in ("fused", "fused_interp", "fused_smooth"):
        head = [l for l in (fix / f"clip.{v}.txt").read_text().splitlines() if l.startswith("# track:")][0]
        assert "sigma_acc 0.45 mm/frame^2 | sigma_s 2.3 mm |" in head, v
    for v in ("fused", "fused_interp"):                        # raw and interp do not depend on the smoother
        a = [l for l in (est / f"clip.{v}.txt").read_text().splitlines() if not l.startswith("#")]
        b = [l for l in (fix / f"clip.{v}.txt").read_text().splitlines() if not l.startswith("#")]
        assert a == b, v


def test_sigma_s_m_is_checked():
    with pytest.raises(ValueError, match="sigma_s_m"):
        P.Params(sigma_s_m=0.0)


def test_the_column_line_keeps_the_released_names_for_the_neutral_code_names():
    """The column line is a data line of the released files: it must not change when the code renames a column."""
    assert TW.HEADER == ("id frame t_s x/m y/m z/m observed source miss_mm ray_angle_deg view_basler_deg corner_basler "
                         "r_basler reproj_px flags markerID x_floor/m y_floor/m z_floor/m surface_id floor_src")
    names = [TW.FILE_NAMES.get(c, c) for c in TW.COLUMNS]
    assert TW.HEADER.split() == names[:15] + ["markerID"] + names[15:]


def test_a_fisheye_calibration_is_refused_before_anything_is_posed(tmp_path):
    """The trajectory step poses with the pinhole model (the orientation step uses `solvePnPGeneric`), so a fisheye
    calibration is refused at the top of `metric_step`, before the geometry or any file is written."""
    calib = make_fisheye_calib()
    c = _cache(tmp_path / "f.npz", _calib(tmp_path), _walk())
    with pytest.raises(ValueError, match="fisheye calibration.*pinhole"):
        P.metric_step("clip", c, c, tmp_path / "out", tmp_path / "work", calib=calib, floor=None,
                      camera="cam")
    assert not (tmp_path / "work").exists() and not (tmp_path / "out").exists()


def test_the_orientation_candidates_refuse_a_fisheye_calibration():
    """`marker_orient.candidates` solves with `solvePnPGeneric`, which knows only the pinhole model: a fisheye
    calibration is refused by name rather than giving orientations through the wrong lens model."""
    from arucotrack.geometry import marker_orient
    uv = np.array([[[1000.0, 900.0], [1040.0, 900.0], [1040.0, 940.0], [1000.0, 940.0]]])
    with pytest.raises(ValueError, match="fisheye"):
        marker_orient.candidates(make_fisheye_calib(), uv)


def test_ids_with_too_few_rows_are_counted_in_the_sidecar(tmp_path):
    calib = _calib(tmp_path)
    rows = _walk(n=40) + [(99, 3, np.array([0.0, -1.0, 1.7]))]
    _, out = _run(tmp_path, calib, rows, None, "few", params=P.Params(height_min=None, height_max=None))
    side = json.loads((out / "clip.sidecar.json").read_text())
    assert side["gating"]["too_few_rows"] == {"99": 1}
