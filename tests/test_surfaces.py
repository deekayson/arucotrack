"""Floor projection onto measured walking surfaces (landing + treads)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from arucotrack.geometry import surfaces as SF


def two_steps():
    """Landing z=0 for x<1, tread A z=0.17 for 1<=x<1.3, tread B z=0.34 for 1.3<=x<1.6."""
    sq = lambda x0, x1: [[x0, -0.5], [x1, -0.5], [x1, 0.5], [x0, 0.5]]
    return [
        SF.Surface(0, "landing", np.array(sq(-1.0, 1.0)), (0.0, 0.0, 1.0, 0.0)),
        SF.Surface(1, "tread", np.array(sq(1.0, 1.3)), (0.0, 0.0, 1.0, -0.17)),
        SF.Surface(2, "tread", np.array(sq(1.3, 1.6)), (0.0, 0.0, 1.0, -0.34)),
    ]


def test_vertical_drop_onto_each_surface():
    xyz = np.array([[0.0, 0.0, 1.7], [1.1, 0.1, 1.87], [1.5, -0.2, 2.04]])
    out, sid = SF.project_to_surfaces(xyz, two_steps())
    assert sid.tolist() == [0, 1, 2]
    assert np.allclose(out[:, :2], xyz[:, :2])
    assert np.allclose(out[:, 2], [0.0, 0.17, 0.34])


def test_tilted_plane_uses_plane_equation():
    s = [SF.Surface(0, "landing", np.array([[0, 0], [2, 0], [2, 2], [0, 2.0]]), (0.1, 0.0, 1.0, -0.05))]
    out, sid = SF.project_to_surfaces(np.array([[1.0, 1.0, 2.0]]), s)
    assert sid[0] == 0 and out[0, 2] == pytest.approx(-0.1 * 1.0 + 0.05)


def test_outside_all_polygons_is_nan_and_minus_one():
    out, sid = SF.project_to_surfaces(np.array([[5.0, 5.0, 1.7]]), two_steps())
    assert sid[0] == -1 and np.isnan(out).all()


def test_point_below_every_plane_is_outside():
    out, sid = SF.project_to_surfaces(np.array([[1.5, 0.0, 0.10]]), two_steps())
    assert sid[0] == -1 and np.isnan(out).all()


def test_overlap_takes_highest_plane_below_point():
    # A landing under a mezzanine: same xy, planes at 0 and 2.5.
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    s = [SF.Surface(0, "landing", sq, (0, 0, 1, 0.0)), SF.Surface(1, "landing", sq, (0, 0, 1, -2.5))]
    _, sid = SF.project_to_surfaces(np.array([[0.5, 0.5, 1.7], [0.5, 0.5, 4.2]]), s)
    assert sid.tolist() == [0, 1]


def test_tread_edge_belongs_to_a_surface_and_riser_is_deterministic():
    # x exactly on the nosing line x=1.0 and on x=1.3 must land on some surface, not NaN.
    xyz = np.array([[1.0, 0.0, 1.9], [1.3, 0.0, 2.1]])
    _, sid = SF.project_to_surfaces(xyz, two_steps())
    assert (sid >= 0).all()


def test_nan_input_is_outside():
    out, sid = SF.project_to_surfaces(np.array([[np.nan, 0, 1.7]]), two_steps())
    assert sid[0] == -1 and np.isnan(out).all()


def test_empty_input():
    out, sid = SF.project_to_surfaces(np.zeros((0, 3)), two_steps())
    assert out.shape == (0, 3) and sid.shape == (0,)


def test_surfaces_json_loads(tmp_path):
    doc = {"frame": "world_m", "source_mesh": "x.ply", "surfaces": [
        {"id": 3, "kind": "tread", "polygon_xy": [[0, 0], [1, 0], [1, 1], [0, 1]],
         "plane": [0, 0, 1, -0.17], "z_mean": 0.17, "rms_mm": 2.0, "n_pts": 100}]}
    p = tmp_path / "g.json"
    p.write_text(json.dumps(doc))
    surf = SF.load_surfaces(p)
    assert surf[0].id == 3 and surf[0].kind == "tread"


def test_loader_rejects_wrong_frame(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"frame": "mm", "surfaces": []}))
    with pytest.raises(ValueError):
        SF.load_surfaces(p)
