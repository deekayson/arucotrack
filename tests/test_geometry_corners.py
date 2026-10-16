"""`geometry.corner_handling`: refined caches must record their corner convention, and per-row flags derive does not
apply are reported."""
import numpy as np
import pytest
from arucotrack.geometry.pose import corner_handling


def _z(tmp_path, **arrays):
    p = tmp_path / "c.npz"
    np.savez(p, det_corners=np.zeros((2, 4, 2), np.float32), **arrays)
    return np.load(p)


def test_a_refined_cache_without_its_corner_convention_is_refused(tmp_path):
    z = _z(tmp_path, det_refine_rms_px=np.zeros(2))
    with pytest.raises(ValueError, match="c.npz.*corner_convention"):
        corner_handling(z, {}, "c.npz")
    # a centre-convention refined cache and a detector cache are posed as stored
    assert corner_handling(z, {"corner_convention": "centre"}, "c.npz") == {"corner_convention": "centre",
                                                                             "unused_flags": []}
    assert corner_handling(_z(tmp_path), {}, "c.npz") == {"corner_convention": None, "unused_flags": []}
    z.close()


def test_flags_that_derive_does_not_apply_are_reported(tmp_path):
    z = _z(tmp_path, det_refine_rms_px=np.zeros(2), det_refine_used=np.array([True, False]))
    note = corner_handling(z, {"corner_convention": "centre"}, "c.npz")
    assert note["unused_flags"] == ["det_refine_used"]
    z.close()
    z = _z(tmp_path, det_refine_rms_px=np.zeros(2), det_refine_used=np.array([40, 40], np.int32))   # pre-gate count
    assert corner_handling(z, {"corner_convention": "centre"}, "c.npz")["unused_flags"] == []
    z.close()
