"""The fused_interp gap fill (`pipeline.interp_cols`), the packet frame rate and the number format of the files."""
import numpy as np

from arucotrack import pipeline as P
from arucotrack.io import trajwrite as TW


def test_num_never_writes_negative_zero_or_minus_nan():
    assert TW.num(-0.00001, 4) == "0.0000" and TW.num(float("nan"), 3) == "nan" and TW.num(-1.5, 2) == "-1.50"
    assert TW.num(float("-inf"), 3) == "nan"


def test_framerate_from_pts_is_packet_based():
    pts = 0.253 + np.r_[np.arange(100), np.arange(102, 200)] / 59.84     # two dropped packets
    assert abs(P.framerate_from_pts(pts) - (len(pts) - 1) / (pts[-1] - pts[0])) < 1e-12


def _cols(rows):
    """rows: (id, frame, x, y, z) -> measured fused columns."""
    c = P.empty_cols()
    n = len(rows)
    a = np.array(rows, float)
    c.update(id=a[:, 0].astype(int), frame=a[:, 1].astype(int), x=a[:, 2], y=a[:, 3], z=a[:, 4],
             observed=np.ones(n, int), source=np.ones(n, int), corner=np.array(["refined"] * n, object),
             flags=np.array(["-"] * n, object), surface_id=np.full(n, -1), floor_src=np.zeros(n, int))
    for col in TW.FLOAT_COLS:
        if col not in ("x", "y", "z"):
            c[col] = np.full(n, 0.5)
    c["t_s"] = a[:, 1] / 60.0
    return c


def test_interp_cols_fills_gaps_up_to_the_cap_linearly_with_packet_times():
    pts = 0.25 + np.r_[np.arange(40), np.arange(41, 120)] / 59.8          # one dropped packet
    fr = [3, 4, 9, 40, 41, 75]          # gap 5 filled, gaps 31 and 34 exceed the cap and stay
    rows = [(17, f, 0.01 * f, -0.02 * f, 1.7 + 0.001 * f) for f in fr] + [(18, 10, 1.0, 0.0, 1.2), (18, 12, 1.2, 0.2, 1.2)]
    rows += [(19, 36, 3.6, 0.0, 1.5), (19, 44, 4.4, 0.8, 1.5)]          # gap 8 spanning the dropped packet at frame 40
    out = P.interp_cols(_cols(rows), pts, cap=30)
    assert np.all(np.diff(np.lexsort((out["frame"], out["id"]))) == 1)          # sorted by (id, frame)
    filled = out["observed"] == 0
    got = sorted(zip(out["id"][filled].tolist(), out["frame"][filled].tolist()))
    assert got == [(17, 5), (17, 6), (17, 7), (17, 8), (18, 11)] + [(19, g) for g in range(37, 44)]
    f = out["frame"][filled]
    assert np.allclose(out["x"][filled][:4], 0.01 * f[:4]) and np.allclose(out["z"][filled][:4], 1.7 + 0.001 * f[:4])
    at18 = (out["id"] == 18) & (out["frame"] == 11)
    assert np.allclose([out["x"][at18][0], out["y"][at18][0], out["z"][at18][0]], [1.1, 0.1, 1.2])
    s19 = filled & (out["id"] == 19)                  # weights follow the frame index, not the packet times
    assert np.allclose(out["x"][s19], 0.1 * out["frame"][s19]) and np.allclose(out["y"][s19], 0.1 * (out["frame"][s19] - 36))
    assert np.allclose(out["t_s"][filled], pts[f])
    assert (out["source"][filled] == 0).all() and set(out["flags"][filled]) == {"interp"}
    assert np.isnan(out["reproj_px"][filled]).all() and (out["corner"][filled] == "-").all()
    meas = ~filled
    assert sorted(zip(out["id"][meas].tolist(), out["frame"][meas].tolist())) == sorted((r[0], r[1]) for r in rows)
    assert (out["reproj_px"][meas] == 0.5).all()                               # measured rows unchanged


def test_the_z_gate_runs_before_the_one_pose_per_frame_dedupe(tmp_path):
    """A false quad posed outside the world-z gate that comes first in its frame must not displace the real head
    row of the same id in that frame: the gate removes it, then one row per (id, frame) is kept."""
    from arucotrack.geometry.pose import derive
    from tests.test_pipeline import _cache, _calib, _walk
    calib = _calib(tmp_path)
    walk = _walk()
    false = [(17, f, np.array([0.2, -2.0 + 0.012 * f, -3.5])) for f in (30, 31, 32)]
    rows = []
    for r in walk:
        if r[0] == 17 and r[1] in (30, 31, 32):
            rows.append(false[r[1] - 30])                    # the false row first in cache order
        rows.append(r)
    c = _cache(tmp_path / "c.npz", calib, rows)
    geom = derive(c, calib, out=tmp_path / "c.geom.npz")
    cols, info = P.measured_rows(c, geom, calib, P.Params())
    assert info["z_gated"] == {"17": 3} and info["duplicates"] == {}
    k = (cols["id"] == 17) & np.isin(cols["frame"], [30, 31, 32])
    assert k.sum() == 3 and np.allclose(cols["z"][k], 1.7, atol=0.05)
