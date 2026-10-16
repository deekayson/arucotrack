"""What the refined-corner cache records about its inputs: the border constant (only a number that is passed counts,
nothing is read from beside the cache) and the camera name (the configuration's, else the one the detection cache
records)."""
from __future__ import annotations

import pytest

from arucotrack.refinement import refine_cache


def test_the_camera_is_the_override_else_the_recorded_one_else_none():
    """Nothing is inferred from file names or paths: a camera name in a relpath is not read."""
    prov = {"input": {"relpath": "raw/day_2/cam_overhead/clips/clip_03.mkv"}}
    assert refine_cache.resolve_camera(prov) == ""
    assert refine_cache.resolve_camera(prov, "cam1") == "cam1"
    assert refine_cache.resolve_camera({**prov, "camera": "cam2"}) == "cam2"
    assert refine_cache.resolve_camera({**prov, "camera": "cam2"}, "cam1") == "cam1"


def test_resolve_border_cells_takes_a_number_and_refuses_anything_else(tmp_path):
    v, blk = refine_cache.resolve_border_cells("0.9")
    assert v == 0.9 and blk == {"border_cells": 0.9, "border_source": "override"}
    (tmp_path / "clip_01.border_calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="not a number"):           # a file is never read
        refine_cache.resolve_border_cells(str(tmp_path / "clip_01.border_calibration.json"))
