"""The camera model a run poses with: `Calib`, read from a calibration file (`Calib.from_json`, format in
docs/CALIBRATION_FILE.md), and the check that a cache is posed with its own camera's calibration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import calibration_file

# Pinhole inverse (`Calib.normalise`): at most 100 iterations, stop once the re-projection error is below
# 1e-9 px (OpenCV measures the EPS criterion in distorted pixels, and 1e-9 px is ~1.6e-12 normalised at a focal length of about 640 px).
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)


@dataclass(frozen=True)
class Calib:
    K: np.ndarray
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    Rt: np.ndarray
    cam: np.ndarray          # camera centre, world
    image_size: tuple[int, int]
    source: str
    sha256: str
    # Side of the printed black square, metres. There is no default: every pose needs the user's own print size
    # (`marker_side`).
    marker_m: float | None = None
    # False when no pose is published: side/tilt/residual/ambiguity stay valid,
    # world xyz does not.
    has_extrinsics: bool = True
    # "pinhole" (OpenCV k1k2p1p2[k3...]) or "fisheye" (Kannala-Brandt, 4 coeffs). Only a pinhole calibration is
    # posed: a fisheye one is read so that it can be refused with a clear message (`fisheye`).
    model: str = "pinhole"
    # The camera this calibration belongs to ("" = unknown). `check_calib_for_cache` refuses a cache that records
    # another camera.
    camera: str = ""

    @property
    def fisheye(self) -> bool:
        return self.model == "fisheye"

    def project(self, pts_world) -> np.ndarray:
        """World points (n,3) -> image pixels (n,2), distortion included (pinhole model)."""
        p = np.ascontiguousarray(pts_world, np.float64).reshape(-1, 3)
        uv, _ = cv2.projectPoints(p, self.rvec.reshape(3, 1), self.tvec, self.K, self.D)
        return uv.reshape(-1, 2)

    def normalise(self, uv) -> np.ndarray:
        """Image pixels (n,2) -> ideal normalised coordinates, distortion removed (pinhole model).

        Iterates `cv2.undistortPoints` to convergence (`UNDISTORT_CRITERIA`). Its default of 5 iterations was
        measured to leave 7.2 / 11.1 px (top corners) and 1.6 / 2.4 px (bottom corners) of re-projection error on a
        rational 8-coefficient model of a global-shutter camera. Converged, it is <= 1e-9 px.
        """
        uv = np.asarray(uv, np.float64).reshape(-1, 2)
        return cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.D,
                                   criteria=UNDISTORT_CRITERIA).reshape(-1, 2)

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def half_diag(self) -> float:
        return float(np.hypot(self.image_size[0] / 2, self.image_size[1] / 2))

    @property
    def marker_side(self) -> float:
        """`marker_m`, or ValueError when no marker size was given (a pose cannot be scaled without it)."""
        if self.marker_m is None:
            raise ValueError(f"{self.source}: no marker size given (configuration marker.side_m)")
        return float(self.marker_m)

    def obj(self) -> np.ndarray:
        h = self.marker_side / 2  # aruco corner order: TL, TR, BR, BL in the marker frame
        return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)

    @classmethod
    def from_json(cls, path: str | Path, camera: str = "", marker_m: float | None = None) -> "Calib":
        """A `calibration.json` (pinhole or fisheye) as a `Calib`, the one way a calibration enters the pipeline
        (`camera.calibration`). `camera` overrides the file's `camera_id`. `marker_m` is the printed marker
        side in metres (None: not given, so no pose can be computed with it). The file is checked first
        (`calibration_file.check`, format in `docs/CALIBRATION_FILE.md`), and the ValueError lists every
        problem."""
        try:
            doc = json.loads(Path(path).read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: not JSON ({e})") from None
        problems = calibration_file.check(doc)
        if problems:
            raise ValueError(f"{path}: not a usable calibration file: " + "; ".join(problems))
        c = load_calib_file(Path(path), camera=camera)
        if marker_m is not None:
            import dataclasses
            c = dataclasses.replace(c, marker_m=float(marker_m))
        return c


def load_calib_file(path: Path, camera: str = "") -> Calib:
    """One `calibration.json` that `calibration_file.check` accepted, as a `Calib` (pinhole or fisheye model)."""
    path = Path(path)
    raw = path.read_bytes()
    b = json.loads(raw)
    I = b["intrinsics"]
    is_fisheye = calibration_file.is_fisheye(b)
    size = b.get("image_size") or I.get("image_size")
    K = calibration_file.intrinsic_matrix(b)          # fx, fy, cx, cy, or a 3x3 K
    D = np.asarray(I["dist"], np.float64)
    E = b.get("extrinsics")
    if E:
        rvec, tvec = calibration_file.pose(b)          # rvec / tvec, or R / t
    else:
        # No pose published: image-space and camera-frame quantities (side, tilt,
        # residual, ambiguity) are still valid. World xyz is not, and comes out
        # as camera-frame coordinates, flagged by `has_extrinsics`.
        rvec = np.zeros(3)
        tvec = np.zeros((3, 1))
    R, _ = cv2.Rodrigues(rvec)
    Rt = R.T
    return Calib(K, D, rvec, tvec, R, Rt, (-Rt @ tvec).ravel(),
                 tuple(int(v) for v in size), str(path), hashlib.sha256(raw).hexdigest(),
                 has_extrinsics=bool(E), model="fisheye" if is_fisheye else "pinhole",
                 camera=camera or str(b.get("camera_id") or ""))


def cache_camera(cache_npz: str | Path) -> str:
    """The camera a cache's provenance records (`camera`, written by `refine_cache.run`), "" when it records none."""
    with np.load(Path(cache_npz), allow_pickle=False) as z:
        prov = json.loads(str(z["provenance"])) if "provenance" in z.files else {}
    return str(prov.get("camera") or "")


def check_calib_for_cache(cache_npz: str | Path, calib: Calib) -> None:
    """Refuse posing a cache with another camera's calibration (both name their camera and the names differ)."""
    camera = cache_camera(cache_npz)
    name = Path(cache_npz).name
    if calib.camera and camera and calib.camera != camera:
        raise ValueError(f"{name}: a {camera} cache posed with the {calib.camera} calibration ({calib.source})")
