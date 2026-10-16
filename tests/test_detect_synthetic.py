"""Detection of a rendered marker: known pose in, corners out, bias asserted under a bound.

The marker is placed at an integer position on a quiet background, so this checks
the plumbing and the corner order, not the sub-pixel bias. The sub-pixel bias is
measured with the supersampled renderer in `tests/synthetic.py`.
"""
import cv2
import pytest
import numpy as np

from arucotrack.detection import detectors
from arucotrack.detection.detect import make_params
from arucotrack.provenance import detector_params_dict

SIDE, X0, Y0 = 80, 160, 120
DICTIONARY = cv2.aruco.DICT_ARUCO_MIP_36h12


def _detect(gray):
    """(corners (n,4,2), ids (n,)) of the default multipass detector at scale 1."""
    dets, _ = detectors.MultiPass(scales=(1,)).detect_with_rejected(gray)
    return (np.asarray([d.corners for d in dets], np.float32).reshape(-1, 4, 2),
            np.asarray([d.id for d in dets], np.int32))


def _canvas(marker_id: int) -> np.ndarray:
    dic = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    marker = cv2.aruco.generateImageMarker(dic, marker_id, SIDE)
    canvas = np.full((400, 400), 200, np.uint8)  # light background = the quiet zone
    canvas[Y0:Y0 + SIDE, X0:X0 + SIDE] = marker
    return canvas


def test_detects_id_37_with_subpixel_corners():
    corners, ids = _detect(_canvas(37))
    assert ids.tolist() == [37]
    q = corners[0]
    # aruco order: top-left, top-right, bottom-right, bottom-left. Corner k sits on pixel edges
    truth = np.array([[X0, Y0], [X0 + SIDE, Y0], [X0 + SIDE, Y0 + SIDE], [X0, Y0 + SIDE]], np.float32)
    err = np.linalg.norm(q - truth, axis=1)
    # generateImageMarker's outer edge sits on pixel boundaries, SUBPIX refines to the edge centre,
    # so half a pixel of systematic offset is expected. Anything beyond ~1 px is a real defect.
    assert err.max() < 1.0, f"corner error {err}"
    side = np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)])
    assert abs(side - SIDE) < 1.5


def test_blank_frame_yields_nothing():
    corners, ids = _detect(np.full((300, 300), 128, np.uint8))
    assert len(ids) == 0 and corners.shape == (0, 4, 2)


def test_params_are_recorded_in_full():
    d = detector_params_dict(make_params())
    assert d["cornerRefinementMethod"] == cv2.aruco.CORNER_REFINE_SUBPIX
    assert len(d) > 20, "expected the whole parameter set, not a diff"


def test_resolve_threads(monkeypatch):
    import os

    from arucotrack.detection import detect
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4}, raising=False)
    assert detect.resolve_threads(0) == 5
    assert detect.resolve_threads(4) == 4
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    assert detect.resolve_threads(0) == 7
    with pytest.raises(ValueError, match="detect.threads"):
        detect.resolve_threads(-1)


def test_default_threads_is_all_usable_cores():
    from arucotrack import config as C
    assert C.DetectConfig.threads == 0
    assert "all usable cores" in C.markdown() and "shared machine" in C.markdown()


def test_detections_do_not_depend_on_the_thread_count():
    from arucotrack.detection import detect
    out = []
    try:
        for n in (1, 3):
            detect.set_threads(n)
            assert cv2.getNumThreads() == n
            c, ids = _detect(_canvas(37))
            out.append((ids.tobytes(), c.tobytes()))
    finally:
        detect.set_threads(1)
    assert out[0] == out[1] and out[0][0]


def test_peak_rss_is_in_mib_on_linux_and_macos(monkeypatch):
    import resource
    import sys
    from arucotrack.detection import detect

    class _R:
        ru_maxrss = 2 * 2**20

    monkeypatch.setattr(resource, "getrusage", lambda who: _R())
    monkeypatch.setattr(sys, "platform", "linux")
    assert detect.peak_rss_mib() == 2048
    monkeypatch.setattr(sys, "platform", "darwin")
    assert detect.peak_rss_mib() == 2
