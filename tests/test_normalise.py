"""`Calib.normalise` (pinhole) is converged: the rational-8 model of the reference calibration round-trips at the frame corners."""
import cv2
import numpy as np


def _reproject(calib, ideal):
    obj = np.column_stack([ideal, np.ones(len(ideal))]).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), calib.K, calib.D)
    return uv.reshape(-1, 2)


def test_normalise_round_trips_the_whole_frame(pinhole_calib):
    W, H = pinhole_calib.image_size
    us, vs = np.meshgrid(np.linspace(0, W - 1, 60), np.linspace(0, H - 1, 45))
    uv = np.column_stack([us.ravel(), vs.ravel()])
    assert np.abs(_reproject(pinhole_calib, pinhole_calib.normalise(uv)) - uv).max() < 1e-3
    # the test is not vacuous: OpenCV's default 5 iterations miss the top frame corners by pixels
    corners = np.array([[0, 0], [W - 1, 0]], np.float64)
    five = cv2.undistortPoints(corners.reshape(-1, 1, 2), pinhole_calib.K, pinhole_calib.D).reshape(-1, 2)
    assert np.linalg.norm(_reproject(pinhole_calib, five) - corners, axis=1).min() > 5.0
