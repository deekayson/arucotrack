import os

import cv2
import numpy as np
import pytest

from arucotrack.calibration.camera import Calib

from pathlib import Path

TESTS = Path(__file__).resolve().parent
#: The printed marker side of the test scenes, metres (the package has no default marker size).
MARKER_M = 0.1845


def reference_calib() -> Calib:
    """The calibration of an overhead camera (2592 x 1944, rational-8 distortion, nadir pose 3.11 m up), built in
    place. The package has no default calibration. Tests whose expected numbers depend on a real lens model use this one."""
    K = np.array([[6.3799610498934862e+02, 0.0, 1.2968873340534574e+03],
                  [0.0, 6.3799610498934862e+02, 1.0354405812147684e+03], [0.0, 0.0, 1.0]])
    D = np.array([-2.7817364544731416e-01, 1.8490087636073024e-01, 1.6590804391480561e-04, -5.6695434591384449e-04,
                  -6.6770859165282874e-03, -2.7898072216131814e-01, 1.8261440542670138e-01, -4.1716480591309146e-03])
    rvec = np.array([-3.1372665067631260e+00, 5.4480920385895776e-02, -1.0169295596025314e-04])
    tvec = np.array([1.3322921530975326e-03, -1.3312275466528569e+00, 3.1069337575083917e+00]).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return Calib(K, D, rvec, tvec, R, R.T, (-R.T @ tvec).ravel(), (2592, 1944), "reference-overhead (tests/conftest.py)",
                 "1" * 64, marker_m=MARKER_M, camera="cam_overhead")


def _calib(K, D, size, model):
    K = np.asarray(K, np.float64); D = np.asarray(D, np.float64)
    R = np.eye(3)
    return Calib(K=K, D=D, rvec=np.zeros(3), tvec=np.zeros((3, 1)), R=R, Rt=R.T, cam=np.zeros(3),
                 image_size=size, source=f"synthetic-{model}", sha256="0" * 64, marker_m=MARKER_M,
                 has_extrinsics=False, model=model)


def make_fisheye_calib():
    """A fisheye (Kannala-Brandt) calibration: read by the package only to be refused."""
    return _calib([[750.63, 0, 1356.51], [0, 749.20, 1001.82], [0, 0, 1]],
                  [0.407046, -0.097571, 0.065642, -0.025332], (2688, 2016), "fisheye")


@pytest.fixture
def pinhole_calib():
    return _calib([[637.9961, 0, 1296.8873], [0, 637.9961, 1035.4406], [0, 0, 1]],
                  [-0.278174, 0.184901, 0.000166, -0.000567, -0.006677, -0.278981, 0.182614, -0.004172],
                  (2592, 1944), "pinhole")


def marker_image(marker_id: int, side_px: int, border_cells: float | None = None, supersample: int = 8):
    """The marker's black square, `side_px` px across, with a printed border of `border_cells` data-cell
    widths (None: `generateImageMarker`'s exact one-cell border, as the other tests draw it).

    The square spans cells `1 - b` .. `7 + b` of the 8x8 grid, so a cell is `side_px / (6 + 2b)` px. Drawn at
    `supersample` x and area-averaged, so a sub-pixel border lands where it should."""
    if border_cells is None:
        dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
        return cv2.aruco.generateImageMarker(dic, marker_id, side_px)
    from arucotrack.refinement.refine import cell_grid
    g = cell_grid(marker_id)
    n = side_px * supersample
    lo, hi = 1.0 - border_cells, 7.0 + border_cells
    c = lo + (np.arange(n) + 0.5) / n * (hi - lo)               # cell coordinate of each fine sample
    inside = (c >= 1.0) & (c < 7.0)
    ci = np.clip(np.floor(c).astype(int), 0, 7)
    val = g[ci[:, None], ci[None, :]] * (inside[:, None] & inside[None, :])
    fine = (val * 255).astype(np.float64)
    return fine.reshape(side_px, supersample, side_px, supersample).mean(axis=(1, 3)).round().astype(np.uint8)


class PinholeCanvas:
    """An undistorted view of a pinhole camera at its own focal length, covering the whole undistorted frame: a
    marker drawn on it (`make_warped_marker`) is warped into the camera's raw, distorted frame."""

    def __init__(self, calib: Calib):
        self.calib = calib
        W, H = calib.image_size
        t = np.linspace(0.0, 1.0, 400)
        edge = np.concatenate([np.column_stack([t * (W - 1), np.zeros_like(t)]),
                               np.column_stack([t * (W - 1), np.full_like(t, H - 1.0)]),
                               np.column_stack([np.zeros_like(t), t * (H - 1)]),
                               np.column_stack([np.full_like(t, W - 1.0), t * (H - 1)])])
        ideal = calib.normalise(edge)
        fx, fy = calib.K[0, 0], calib.K[1, 1]
        lo, hi = ideal.min(0), ideal.max(0)
        self.K_rect = np.array([[fx, 0.0, -lo[0] * fx], [0.0, fy, -lo[1] * fy], [0.0, 0.0, 1.0]])
        self.size = (int(np.ceil((hi[0] - lo[0]) * fx)) + 1, int(np.ceil((hi[1] - lo[1]) * fy)) + 1)

    def to_rect(self, uv):
        ideal = self.calib.normalise(np.asarray(uv, np.float64).reshape(-1, 2))
        return ideal * np.diag(self.K_rect)[:2] + self.K_rect[:2, 2]

    def to_raw(self, p):
        p = np.asarray(p, np.float64).reshape(-1, 2)
        ideal = (p - self.K_rect[:2, 2]) / np.diag(self.K_rect)[:2]
        obj = np.column_stack([ideal, np.ones(len(ideal))]).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.calib.K, self.calib.D)
        return uv.reshape(-1, 2)


def make_warped_marker(rect, marker_id=37, side_px=160, centre_rect=None, bg=200, border_cells=None):
    """Draw a marker on a `PinholeCanvas` and warp it into a raw frame through the inverse map.
    `border_cells`: the printed border (`marker_image`), None draws the exact one-cell border."""
    W, H = rect.size
    canvas = np.full((H, W), bg, np.uint8)
    m = marker_image(marker_id, side_px, border_cells)
    cx, cy = centre_rect if centre_rect is not None else (W // 2, H // 2)
    x0, y0 = int(cx - side_px // 2), int(cy - side_px // 2)
    canvas[y0:y0 + side_px, x0:x0 + side_px] = m
    truth = np.array([[x0, y0], [x0 + side_px, y0], [x0 + side_px, y0 + side_px], [x0, y0 + side_px]], np.float64) - 0.5
    Wr, Hr = rect.calib.image_size
    # Only the patch of the raw frame that can see the marker is mapped; the rest is `bg`.
    margin = 30
    t = np.linspace(0.0, 1.0, 41)
    lo = np.array([x0 - margin, y0 - margin], np.float64)
    hi = np.array([x0 + side_px + margin, y0 + side_px + margin], np.float64)
    ring = np.concatenate([np.column_stack([lo[0] + t * (hi[0] - lo[0]), np.full_like(t, lo[1])]),
                           np.column_stack([lo[0] + t * (hi[0] - lo[0]), np.full_like(t, hi[1])]),
                           np.column_stack([np.full_like(t, lo[0]), lo[1] + t * (hi[1] - lo[1])]),
                           np.column_stack([np.full_like(t, hi[0]), lo[1] + t * (hi[1] - lo[1])])])
    rr = rect.to_raw(ring)
    u0 = max(int(np.floor(rr[:, 0].min())) - 2, 0); u1 = min(int(np.ceil(rr[:, 0].max())) + 3, Wr)
    v0 = max(int(np.floor(rr[:, 1].min())) - 2, 0); v1 = min(int(np.ceil(rr[:, 1].max())) + 3, Hr)
    us, vs = np.meshgrid(np.arange(u0, u1, dtype=np.float64), np.arange(v0, v1, dtype=np.float64))
    rc = rect.to_rect(np.column_stack([us.ravel(), vs.ravel()]))
    mx = np.nan_to_num(rc[:, 0], nan=-1).reshape(us.shape).astype(np.float32)
    my = np.nan_to_num(rc[:, 1], nan=-1).reshape(us.shape).astype(np.float32)
    patch = cv2.remap(canvas, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
    raw = np.full((Hr, Wr), bg, np.uint8)
    raw[v0:v1, u0:u1] = patch
    return raw, truth


def video_from_frames(path, frames, fps=30):
    """An mp4v clip of grey frames (at least three, repeating the given ones: a timing needs two packets)."""
    h, w = frames[0].shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    for f in (frames * 3)[:max(len(frames), 3)]:
        vw.write(f)
    vw.release()


# ------------------------------------------------------------------ calibration documents and refined-cache stand-ins
NADIR = np.diag([1.0, -1.0, -1.0])
OVERHEAD_C = [-0.05, -1.30, 3.10]
FISHEYE_C = [-0.10, 0.01, 2.80]


def _pose(R, C):
    rvec, _ = cv2.Rodrigues(R)
    return {"convention": "cam_from_world", "rvec": rvec.ravel().tolist(), "tvec": (-R @ np.asarray(C, float)).tolist(),
            "R": R.tolist(), "camera_centre_world_m": [float(v) for v in C], "camera_height_m": float(C[2])}


def calib_doc(camera):
    """A calibration file document: "cam_overhead" (rational-8 pinhole) or "cam_fisheye" (Kannala-Brandt)."""
    if camera == "cam_overhead":
        intr = {"fx": 638.05, "fy": 638.09, "cx": 1298.27, "cy": 1041.53,
                "dist": [-0.2782, 0.1849, 0.000166, -0.000567, -0.006677, -0.2790, 0.1826, -0.004172]}
        return {"camera_id": camera, "method": "extrinsic_fit", "model": "FULL_OPENCV",
                "image_size": [2592, 1944], "intrinsics": intr, "extrinsics": _pose(NADIR, OVERHEAD_C)}
    intr = {"fx": 700.0, "fy": 698.5, "cx": 1356.5, "cy": 1001.8, "dist": [0.407, -0.098, 0.066, -0.025]}
    return {"camera_id": camera, "method": "extrinsic_fit", "model": "OPENCV_FISHEYE",
            "image_size": [2688, 2016], "intrinsics": intr, "extrinsics": _pose(NADIR, FISHEYE_C)}


def marker_quad(X, tilt_deg=1.0):
    """World corners of a marker of side `MARKER_M` centred at `X`, tilted about x."""
    a = np.radians(tilt_deg)
    Rm = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    h = MARKER_M / 2
    return np.asarray(X, float) + np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]]) @ Rm.T


def refined_cache_stub(path, camera, corners, *, seq=None, used=None):
    """A refined-cache stand-in whose provenance records `camera` (as `refine_cache.run` writes it, "" = none)."""
    import json
    n = len(corners)
    seq = np.arange(n) if seq is None else np.asarray(seq)
    prov = {"camera": camera or None, "input": {"relpath": "clip_01.x"},
            "corner_convention": "centre", "environment": {"decoder": "x", "pixel_pipeline": "y"}, "parameters": {}}
    np.savez(path, det_seq=seq, det_pts=seq / 60.0, det_id=np.full(n, 17, np.int32),
             det_corners=np.asarray(corners, np.float32), frame_pts_all=np.arange(int(seq.max()) + 1) / 60.0,
             det_refine_used=np.ones(n, bool) if used is None else np.asarray(used, bool),
             det_refine_gate=np.zeros(n, np.uint8), provenance=np.array(json.dumps(prov)))
    return path


def read_trajectory(path) -> dict:
    """The columns of a trajectory file (`trajwrite.write_interface`), by their code names."""
    from arucotrack.io import trajwrite as TW
    names = TW.HEADER.split()
    code_name = {v: k for k, v in TW.FILE_NAMES.items()}
    rows = [ln.split() for ln in Path(path).read_text().splitlines()
            if ln and not ln.startswith("#") and not ln.startswith("id ")]
    cols = {}
    for j, name in enumerate(names):
        key = code_name.get(name, name)
        raw = [r[j] for r in rows]
        if key in TW.INT_COLS or key == "markerID":
            cols[key] = np.array([int(v) for v in raw], int)
        elif key in TW.STR_COLS:
            cols[key] = np.array(raw, dtype=object)
        else:
            cols[key] = np.array([float(v) for v in raw], float)
    return cols


#: OpenCV threads in every test: the CPUs this process may use (all of them where the platform cannot say), at most
#: 8 (the suite may run on a shared machine). Set before each test, so that a
#: test which changes it (a run, the thread-count test) does not change the speed of the tests after it.
TEST_THREADS = min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1))
cv2.setNumThreads(TEST_THREADS)


@pytest.fixture(autouse=True)
def _opencv_threads():
    cv2.setNumThreads(TEST_THREADS)
    yield
    cv2.setNumThreads(TEST_THREADS)          # module fixtures set up between tests run with it too


@pytest.fixture
def caplog_loguru():
    """The loguru messages at WARNING and above logged during the test, as a list of strings."""
    from loguru import logger
    lst: list = []
    sink = logger.add(lst.append, level="WARNING")
    yield lst
    logger.remove(sink)


def _calib_doc(extrinsics: bool) -> dict:
    d = {"camera_id": "cam", "model": "FULL_OPENCV", "image_size": [320, 240],
         "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0, "dist": [-0.1, 0.05, 0.0, 0.0, 0.0]}}
    if extrinsics:
        d["extrinsics"] = {"rvec": [-3.1373, 0.0545, -0.0001], "tvec": [0.0013, -1.3312, 3.1069]}
    return d


@pytest.fixture
def calib_with_extrinsics(tmp_path):
    """Path of a small calibration JSON with intrinsics and a pose (a world frame)."""
    import json
    p = tmp_path / "calibration_pose.json"
    p.write_text(json.dumps(_calib_doc(True)))
    return p


@pytest.fixture
def calib_intrinsics_only(tmp_path):
    """Path of a small calibration JSON with intrinsics only (positions come out in the camera frame)."""
    import json
    p = tmp_path / "calibration_intrinsics.json"
    p.write_text(json.dumps(_calib_doc(False)))
    return p


@pytest.fixture
def blank_video(tmp_path):
    """`blank.mp4`: 10 grey 320x240 frames, no marker. Skips when this build cannot write and read mp4."""
    from arucotrack.io import video
    p = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    if not w.isOpened():
        pytest.skip("cv2.VideoWriter cannot write mp4v here")
    for _ in range(10):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()
    if not p.is_file() or p.stat().st_size == 0:
        pytest.skip("cv2.VideoWriter wrote no file")
    try:
        video.probe(p)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"the blank video is not readable here: {e}")
    return p


#: The marker of the synthetic end-to-end clip (`marker_video`): id, printed side (m), height of its centre (m).
E2E_ID, E2E_SIDE_M, E2E_Z_M = 37, 0.5, 1.7


def write_gray_video(path, frames, fps=10):
    """Encode grey uint8 frames losslessly (x264 -qp 0) with ffmpeg. Skips when ffmpeg or libx264 is missing."""
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not on the PATH")
    h, w = frames[0].shape
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}",
                        "-r", str(fps), "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)],
                       input=b"".join(np.ascontiguousarray(f, np.uint8).tobytes() for f in frames),
                       capture_output=True)
    if p.returncode:
        pytest.skip(f"ffmpeg cannot write a lossless h264 clip here: {p.stderr.decode()[-200:]}")
    return path


@pytest.fixture
def marker_video(tmp_path, calib_with_extrinsics):
    """`walk.mp4`: 8 frames of 320x240 in which one marker (`E2E_ID`, side `E2E_SIDE_M`) moves 2 cm per frame at
    height `E2E_Z_M` below the camera of `calib_with_extrinsics`, rendered through its camera model."""
    from arucotrack.calibration.camera import Calib
    from tests.synthetic import render
    calib = Calib.from_json(calib_with_extrinsics, marker_m=E2E_SIDE_M)
    R, t = calib.R, np.asarray(calib.tvec, float).reshape(3)
    frames = []
    for k in range(8):
        X = np.array([calib.cam[0] - 0.07 + 0.02 * k, calib.cam[1], E2E_Z_M])
        rv = cv2.Rodrigues(R)[0].ravel()                 # marker frame = world frame (face up)
        r = render(calib, E2E_ID, rv, R @ X + t, bg=200.0, rng=np.random.default_rng(k))
        img = np.full((240, 320), 200, np.uint8)
        x0, y0 = int(r.origin[0]), int(r.origin[1])
        P = r.patch.shape[0]
        img[y0:y0 + P, x0:x0 + P] = r.patch
        frames.append(img)
    return write_gray_video(tmp_path / "walk.mp4", frames)
